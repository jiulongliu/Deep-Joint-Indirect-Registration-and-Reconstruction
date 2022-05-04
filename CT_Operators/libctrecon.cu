// *----------------------------------------------
// 	Author Contact Information:
// 	Jiulong Liu
// 	matliuj@nus.edu.sg || jiu.liu@gmail.com
// 	Department of Mathematics, National University of Singapore

//
// If you find this code useful, you may cite the following reference:
//  



#include <math.h>
#include <malloc.h>

#include <cublas.h>
#include <cublas_v2.h>
//#include <cufftXt.h>
#define ABS(a) (a>0?a:-(a))
#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)
#define BLOCK_SIZE_x 16
#define BLOCK_SIZE_y 16





#include <math.h>
#include <malloc.h>
#define ABS(a) (a>0?a:-(a))

#define BLOCK_SIZE_x 16
#define BLOCK_SIZE_y 16

extern "C" void Ax_fan_mf_gpu_new(float *X,float *y,float SO,float OD,float scale,int nx,int ny,int nv,float *sd_phi,int nd,float *y_det,int *id_X,int nt);
extern "C" void Atx_fan_mf_gpu_new(float *x,float *Y,float SO,float OD,float scale,int nx,int ny,float *sd_phi,int nd,float *y_det,float dy_det,float y_os,int nt,int *id_Y,int *Nv,int tmp_size,int nv,float *cos_phi,float *sin_phi,float *cos_det,float *sin_det);
extern "C" void ctrecon_conjugate_grad(float *x, float *y, float *xp, float SO,float OD,float scale,int nx,int ny,float *sd_phi,int nd, float *y_det,float dy_det,float y_os,int nt,int *id_X,int *id_Y,int *Nv,int tmp_size,int nv,float *cos_phi,float *sin_phi,float *cos_det,float *sin_det, float mu, float rho, int CG_maxit, float CG_tol0, float lambda);
extern "C" void ctrecon_admm(float *x, float *y, float *xp, float SO,float OD,float scale,int nx,int ny,float *sd_phi,int nd, float *y_det,float dy_det,float y_os,int nt,int *id_X,int *id_Y,int *Nv,int tmp_size,int nv,float *cos_phi,float *sin_phi,float *cos_det,float *sin_det, float mu, float rho, int CG_maxit, int N_iter, float CG_tol0, float lambda);



__global__ void Ax_fan_mf_gpu_new_kernel(float *X,float *y,float SO,float OD,float scale,int nx,int ny,int nv,float *sd_phi,int nd,float *y_det,int *id_X)
// Please note that this version has O(Nx) per thread, since GPU threads are already saturated.
// O(1) per thread can be achieved by parallelizing the "for" loop here, given sufficient number of GPU threads.
// If you find this code useful, you may cite the following reference:
// H. Gao. "Fast parallel algorithms for the X-ray transform and its adjoint", Medical Physics (2012).
// The full source codes are available at https://sites.google.com/site/fastxraytransform
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx0=threadIdx.x;
	int ty0=threadIdx.y;

	int iv=bx*BLOCK_SIZE_x+tx0;
	int id=by*BLOCK_SIZE_y+ty0;

    if(iv<nv&&id<nd)
    {
		int n,nx2,ny2,ix,iy,c1,c2;
		float *x,cos_phi,sin_phi,x1,y1,x2,y2,xx1,yy1,xx2,yy2,slope,l,d;

		nx2=nx/2;ny2=ny/2;
		n=nx*ny;

		x=&X[id_X[iv]*n];
        cos_phi=(float)cos(sd_phi[iv]);sin_phi=(float)sin(sd_phi[iv]);
        x1=cos_phi*(-SO);
        y1=sin_phi*(-SO);

        x2=cos_phi*OD-sin_phi*y_det[id];
        y2=sin_phi*OD+cos_phi*y_det[id];

        y[iv*nd+id]=0;
        if(ABS(x1-x2)>ABS(y1-y2))
        {   slope=(y2-y1)/(x2-x1);
            for(ix=0;ix<nx;ix++)
            {   xx1=(float)(ix-nx2);xx2=xx1+1;
                if(slope>=0)
                {   yy1=y1+slope*(xx1-x1)+ny2;
                    yy2=y1+slope*(xx2-x1)+ny2;
                }
                else
                {   yy1=y1+slope*(xx2-x1)+ny2;
                    yy2=y1+slope*(xx1-x1)+ny2;
                }
                c1=(int)floor(yy1);
                c2=(int)floor(yy2);

                if(c2==c1)// c1 and c2 differs less than 1
                {   if(c1>=0&&c1<=ny-1)
                    {   d=yy2-yy1;l=(float)sqrt(d*d+1);
                        iy=c1;y[iv*nd+id]+=l*x[iy*nx+ix];
                    }
                }
                else
                {   if(c2>0&&c2<ny)
                    {   d=yy2-yy1;l=(float)sqrt(d*d+1);
                        iy=c1;y[iv*nd+id]+=((c2-yy1)/d)*l*x[iy*nx+ix];
                        iy=c2;y[iv*nd+id]+=((yy2-c2)/d)*l*x[iy*nx+ix];
                    }
                    else
                    {   if(c2==0)
                        {   d=yy2-yy1;l=(float)sqrt(d*d+1);
                            iy=c2;y[iv*nd+id]+=((yy2-c2)/d)*l*x[iy*nx+ix];
                        }
                        if(c2==ny)
                        {   d=yy2-yy1;l=(float)sqrt(d*d+1);
                            iy=c1;y[iv*nd+id]+=((c2-yy1)/d)*l*x[iy*nx+ix];
                        }
                    }
                }
            }
        }
        else
        {   slope=(x2-x1)/(y2-y1);
            for(iy=0;iy<ny;iy++)
            {   yy1=(float)(iy-ny2);yy2=yy1+1;

                if(slope>=0)
                {   xx1=x1+slope*(yy1-y1)+nx2;
                    xx2=x1+slope*(yy2-y1)+nx2;
                }
                else
                {   xx1=x1+slope*(yy2-y1)+nx2;
                    xx2=x1+slope*(yy1-y1)+nx2;
                }
                c1=(int)floor(xx1);
                c2=(int)floor(xx2);

                if(c2==c1)// c1 and c2 differs less than 1
                {   if(c1>=0&&c1<=nx-1)
                    {   d=xx2-xx1;l=(float)sqrt(d*d+1);
                        ix=c1;y[iv*nd+id]+=l*x[iy*nx+ix];
                    }
                }
                else
                {   if(c2>0&&c2<nx)
                    {   d=xx2-xx1;l=(float)sqrt(d*d+1);
                        ix=c1;y[iv*nd+id]+=((c2-xx1)/d)*l*x[iy*nx+ix];
                        ix=c2;y[iv*nd+id]+=((xx2-c2)/d)*l*x[iy*nx+ix];
                    }
                    else
                    {   if(c2==0)
                        {   d=xx2-xx1;l=(float)sqrt(d*d+1);
                            ix=c2;y[iv*nd+id]+=((xx2-c2)/d)*l*x[iy*nx+ix];
                        }
                        if(c2==ny)
                        {   d=xx2-xx1;l=(float)sqrt(d*d+1);
                            ix=c1;y[iv*nd+id]+=((c2-xx1)/d)*l*x[iy*nx+ix];
                        }
                    }
                }
            }
        }
        y[iv*nd+id]*=scale;
    }
}


void Ax_fan_mf_gpu_new(float *X,float *y,float SO,float OD,float scale,int nx,int ny,int nv,float *sd_phi,int nd,float *y_det,int *id_X,int nt)
// A new method for computing the X-ray transform (infinitely-narrow beam)
// The algorithm details are available in
// H. Gao. "Fast parallel algorithms for the X-ray transform and its adjoint", Medical Physics (2012).
{   float *y_d,*X_d,*sd_phi_d,*y_det_d;
	int *id_X_d;

	cudaMalloc(&y_d,nv*nd*sizeof(float));
	cudaMalloc(&X_d,nx*ny*nt*sizeof(float));
	cudaMalloc(&sd_phi_d,nv*sizeof(float));
	cudaMalloc(&y_det_d,nd*sizeof(float));
	cudaMalloc(&id_X_d,nv*sizeof(int));

	cudaMemcpy(X_d,X,nx*ny*nt*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(sd_phi_d,sd_phi,nv*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(y_det_d,y_det,nd*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(id_X_d,id_X,nv*sizeof(int),cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_t((nv+dimBlock.x-1)/dimBlock.x,(nd+dimBlock.y-1)/dimBlock.y);

	Ax_fan_mf_gpu_new_kernel<<<dimGrid_t, dimBlock>>>(X_d,y_d,SO,OD,scale,nx,ny,nv,sd_phi_d,nd,y_det_d,id_X_d);

	cudaMemcpy(y,y_d,nv*nd*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(y_d);cudaFree(X_d);cudaFree(sd_phi_d);cudaFree(y_det_d);cudaFree(id_X_d);
}

__device__ float find_l(float a,float b,float c,float x,float y)
// A method for computing the intersecting length of a pixel with a infinitely-narrow beam
// The algorithm details are available in
// H. Gao. "Fast parallel algorithms for the X-ray transform and its adjoint", Medical Physics (2012).
{   float l=0,tmp,d,d0,dmax,a2,b2,tmpx,tmpy;

    a2=ABS(a);b2=ABS(b);
    tmpx=a2/2;tmpy=b2/2;
    dmax=tmpx+tmpy;
    tmp=c-a*x-b*y;
    d=ABS(tmp);

    if(d<dmax)
    {   tmp=tmpx-tmpy;
        d0=ABS(tmp);
        if(tmpx<tmpy)
        {tmp=(float)1.0/b2;}
        else
        {tmp=(float)1.0/a2;}
        if(d<=d0)
        {l=tmp;}
        else
        {l=(dmax-d)/(a2*b2);}
    }

    return l;
}

__global__ void Atx_fan_mf_gpu_new_kernel(float *x,float *Y,float SO,float OD,float scale,int nx,int ny,float *sd_phi,int nd,float *y_det,float dy_det,float y_os,
int nt,int *id_Y,int *Nv,int tmp_size,float *cos_phi,float *sin_phi,float *cos_det,float *sin_det)
// Please note that this version has O(Nv) per thread, since GPU threads are already saturated.
// O(1) per thread can be achieved by parallelizing the "for" loop here, given sufficient number of GPU threads.
{
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;

	int ix=bx*BLOCK_SIZE_x+tx;
	int iy2=by*BLOCK_SIZE_y+ty;

    if(ix<nx&&iy2<ny*nt)
    {
		int nx2,ny2,nd2,it,iv,id,iy,nd_min,nd_max,j;
		float *y,xc,yc,xr,yr,SD,Px,Py,Pd,l,tmp;

		SD=SO+OD;

		nx2=nx/2;ny2=ny/2;nd2=nd/2;
		it=(int)floor((float)iy2/(float)ny);
		iy=iy2-it*ny;

		yc=(float)(iy+0.5-ny2);
		xc=(float)(ix+0.5-nx2);

        x[iy2*nx+ix]=0;
        for(j=0;j<Nv[it];j++)
        {   iv=id_Y[it*tmp_size+j];
            y=&Y[iv*nd];

            xr=cos_phi[iv]*xc+sin_phi[iv]*yc;
            yr=-sin_phi[iv]*xc+cos_phi[iv]*yc;
            tmp=SD/((xr+SO)*dy_det);
            nd_max=(int)floor((yr+1)*tmp-y_os+nd2);
            nd_min=(int)floor((yr-1)*tmp-y_os+nd2);
            for(id=MAX(0,nd_min);id<=MIN(nd_max,nd-1);id++)
            {   Px=-(sin_det[id]*cos_phi[iv]+cos_det[id]*sin_phi[iv]);
                Py=cos_det[id]*cos_phi[iv]-sin_det[id]*sin_phi[iv];
                Pd=SO*sin_det[id];
                l=find_l(Px,Py,Pd,xc,yc);
                if(l>0)
                {x[iy2*nx+ix]+=l*y[id];}
            }
        }
        x[iy2*nx+ix]*=scale;
    }
}

void Atx_fan_mf_gpu_new(float *x,float *Y,float SO,float OD,float scale,int nx,int ny,float *sd_phi,int nd,float *y_det,float dy_det,float y_os,
int nt,int *id_Y,int *Nv,int tmp_size,int nv,float *cos_phi,float *sin_phi,float *cos_det,float *sin_det)
// A new method for computing the adjoint X-ray transform (infinitely-narrow beam)
// The algorithm details are available in
// H. Gao. "Fast parallel algorithms for the X-ray transform and its adjoint", Medical Physics (2012).
{   float *x_d,*Y_d,*sd_phi_d,*y_det_d,*cos_phi_d,*sin_phi_d,*cos_det_d,*sin_det_d;
	int *id_Y_d,*Nv_d;

	cudaMalloc(&cos_phi_d,nv*sizeof(float));cudaMemcpy(cos_phi_d,cos_phi,nv*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&sin_phi_d,nv*sizeof(float));cudaMemcpy(sin_phi_d,sin_phi,nv*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&cos_det_d,nd*sizeof(float));cudaMemcpy(cos_det_d,cos_det,nd*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&sin_det_d,nd*sizeof(float));cudaMemcpy(sin_det_d,sin_det,nd*sizeof(float),cudaMemcpyHostToDevice);

	cudaMalloc(&x_d,nx*ny*nt*sizeof(float));
	cudaMalloc(&Y_d,nv*nd*sizeof(float));cudaMemcpy(Y_d,Y,nv*nd*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&sd_phi_d,nv*sizeof(float));cudaMemcpy(sd_phi_d,sd_phi,nv*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&y_det_d,nd*sizeof(float));cudaMemcpy(y_det_d,y_det,nd*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&id_Y_d,nt*tmp_size*sizeof(int));cudaMemcpy(id_Y_d,id_Y,nt*tmp_size*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&Nv_d,nt*sizeof(int));cudaMemcpy(Nv_d,Nv,nt*sizeof(int),cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_t((nx+dimBlock.x-1)/dimBlock.x,(ny*nt+dimBlock.y-1)/dimBlock.y);
	Atx_fan_mf_gpu_new_kernel<<<dimGrid_t, dimBlock>>>(x_d,Y_d,SO,OD,scale,nx,ny,sd_phi_d,nd,y_det_d,dy_det,y_os,
	nt,id_Y_d,Nv_d,tmp_size,cos_phi_d,sin_phi_d,cos_det_d,sin_det_d);

	cudaMemcpy(x,x_d,nx*ny*nt*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(x_d);cudaFree(Y_d);cudaFree(sd_phi_d);cudaFree(y_det_d);cudaFree(id_Y_d);cudaFree(Nv_d);
    cudaFree(cos_phi_d);cudaFree(sin_phi_d);cudaFree(cos_det_d);cudaFree(sin_det_d);
}


















__global__ void wx_2d_kernel(float *wx,float *x,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx;
		idx=iy*nx+ix;
		if(ix<nx-1)		
		{
			wx[2*idx]=x[idx]-x[idx+1];
		}
		else
		{
			wx[2*idx]=0;
		}
		if(iy<ny-1)		
		{
			wx[2*idx+1]=x[idx]-x[idx+nx];
		}
		else
		{
			wx[2*idx+1]=0;
		}
	}
}
void wx_2d_d(float *wx_d,float *x_d,int nx,int ny)
{   dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_t((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	wx_2d_kernel<<<dimGrid_t, dimBlock>>>(wx_d,x_d,nx,ny);
	cudaThreadSynchronize();			
}

__global__ void wtx_2d_kernel(float *x,float *wx,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx;
		idx=iy*nx+ix;
		
		x[idx]=0;
		if(ix==0)		
		{x[idx]+=wx[2*idx];}
		else
		{	if(ix<nx-1)
			{
				x[idx]+=-wx[2*(idx-1)]+wx[2*idx];
			}
			else
			{
				x[idx]+=-wx[2*(idx-1)];
			}
		}
		if(iy==0)		
		{x[idx]+=wx[2*idx+1];}
		else
		{	if(iy<ny-1)
			{
				x[idx]+=-wx[2*(idx-nx)+1]+wx[2*idx+1];
			}
			else
			{
				x[idx]+=-wx[2*(idx-nx)+1];
			}
		}
		
	}
}
void wtx_2d_d(float *x_d,float *wx_d,int nx,int ny)
{   dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_t((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);			        
	wtx_2d_kernel<<<dimGrid_t, dimBlock>>>(x_d,wx_d,nx,ny);
	cudaThreadSynchronize();			
}

__global__ void shrink_2d_kernel(float *y,float *x,float s,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx;
		float tmp,d;
		idx=2*(iy*nx+ix);		
		tmp=(float)sqrt(x[idx]*x[idx]+x[idx+1]*x[idx+1]);
        if(tmp>s)
        {
			d=(tmp-s)/tmp;
			y[idx]=x[idx]*d;
			y[idx+1]=x[idx+1]*d;}
        else
        {
			y[idx]=0;
			y[idx+1]=0;
		}
	}
}
void shrink_2d_d(float *y,float *x,float s,int nx,int ny)
{   dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_t((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	shrink_2d_kernel<<<dimGrid_t, dimBlock>>>(y,x,s,nx,ny);
	cudaThreadSynchronize();			
}

__global__ void mulC_kernel(float *x,float c,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		x[idx]*=c;
	}
}
__global__ void mulC_plus_kernel(float *y,float *x,float s,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		y[idx]+=s*x[idx];
	}
}

__global__ void plus_kernel(float *y,float *x,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		y[idx]+=x[idx];
	}
}


__global__ void minus_kernel(float *y,float *x,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	
		int idx=(iy*nx+ix);		
		y[idx]-=x[idx];
	}
}


__global__ void initial_kernel(float *y,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	
		int idx=(iy*nx+ix);		
		y[idx]=0.0f;
	}
}

__global__ void equal_kernel(float *y,float *x,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	
		int idx=(iy*nx+ix);		
		y[idx]=x[idx];
	}
}

__global__ void addition_kernel(float *z,float *y, float *x,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		z[idx] = y[idx] + x[idx];
	}
}
__global__ void subtraction_kernel(float *z,float *y, float *x,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		z[idx] = y[idx] - x[idx];
	}
}
__global__ void initialize_kernel(float *y,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		y[idx]= 0.0f;
	}
}


__global__ void equalC_kernel(float *y, float c,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		y[idx] = c;
	}
}







void conjugate_grad_d(float *b_d, float *x_d, float SO,float OD,float scale,int nx,int ny,float *sd_phi,int nd, float *y_det,float dy_det,float y_os,int nt,int *id_X,int *id_Y,int *Nv,int tmp_size,int nv,float *cos_phi,float *sin_phi,float *cos_det,float *sin_det,
  float mu, float rho, int CG_maxit, float CG_tol0,  float *tmpy, float *tmpatax, float *tmpwx,  float *tmpwtwx, float *r,float *p,float *tmpdot)
{

	int iter;	
	float rsold,rsnew,tmpdot_h[1],CG_tol,alpha,beta;
	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_x((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	dim3 dimGrid_t((nv+dimBlock.x-1)/dimBlock.x,(nd+dimBlock.y-1)/dimBlock.y);
	cublasHandle_t h;
	cublasCreate(&h);
	cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
	
	Ax_fan_mf_gpu_new_kernel<<<dimGrid_t, dimBlock>>>(x_d,tmpy,SO,OD,scale,nx,ny,nv,sd_phi,nd,y_det,id_X);cudaThreadSynchronize();
	Atx_fan_mf_gpu_new_kernel<<<dimGrid_x, dimBlock>>>(tmpatax,tmpy,SO,OD,scale,nx,ny,sd_phi,nd,y_det,dy_det,y_os,nt,id_Y,Nv,tmp_size,cos_phi,sin_phi,cos_det,sin_det);cudaThreadSynchronize();
	wx_2d_d(tmpwx,x_d,nx,ny);
	wtx_2d_d(tmpwtwx,tmpwx,nx,ny);	
	mulC_plus_kernel<<<dimGrid_x, dimBlock>>>(tmpatax,tmpwtwx,mu,nx,ny);cudaThreadSynchronize();
	mulC_plus_kernel<<<dimGrid_x, dimBlock>>>(tmpatax,x_d,rho,nx,ny);cudaThreadSynchronize();
	subtraction_kernel<<<dimGrid_x, dimBlock>>>(r,b_d,tmpatax,nx,ny);cudaThreadSynchronize();
	equal_kernel<<<dimGrid_x, dimBlock>>>(p,r,nx,ny);cudaThreadSynchronize();
//	equalC_kernel<<<dimGrid_x, dimBlock>>>(r,(float)0.3,nx,ny);cudaThreadSynchronize(); //will be 0.25 * number of pixels
	cublasSdot(h,nx*ny,r,1,r,1,tmpdot);
	cudaMemcpy(tmpdot_h,tmpdot,1*sizeof(float),cudaMemcpyDeviceToHost);
	rsold=tmpdot_h[0];
	CG_tol=rsold*CG_tol0;
	
	for(iter=0; iter<CG_maxit && rsold>CG_tol;iter++) //&& rsold>CG_tol CG_maxit
	{
		Ax_fan_mf_gpu_new_kernel<<<dimGrid_t, dimBlock>>>(p,tmpy,SO,OD,scale,nx,ny,nv,sd_phi,nd,y_det,id_X);//p
		Atx_fan_mf_gpu_new_kernel<<<dimGrid_x, dimBlock>>>(tmpatax,tmpy,SO,OD,scale,nx,ny,sd_phi,nd,y_det,dy_det,y_os,nt,id_Y,Nv,tmp_size,cos_phi,sin_phi,cos_det,sin_det);
		wx_2d_d(tmpwx,p,nx,ny);//p
		wtx_2d_d(tmpwtwx,tmpwx,nx,ny);	

		mulC_plus_kernel<<<dimGrid_x, dimBlock>>>(tmpatax,tmpwtwx,mu,nx,ny);cudaThreadSynchronize();
		mulC_plus_kernel<<<dimGrid_x, dimBlock>>>(tmpatax,p,rho,nx,ny);cudaThreadSynchronize();
		cublasSdot(h,nx*ny,p,1,tmpatax,1,tmpdot);
		cudaMemcpy(tmpdot_h,tmpdot,1*sizeof(float),cudaMemcpyDeviceToHost);
		rsnew=tmpdot_h[0];
		if(rsnew<CG_tol) break;
		alpha=rsold/rsnew;
		mulC_plus_kernel<<<dimGrid_x, dimBlock>>>(x_d,p,alpha,nx,ny);cudaThreadSynchronize();
		mulC_plus_kernel<<<dimGrid_x, dimBlock>>>(r,tmpatax,-alpha,nx,ny);cudaThreadSynchronize();
		cublasSdot(h,nx*ny,r,1,r,1,tmpdot);
		cudaMemcpy(tmpdot_h,tmpdot,1*sizeof(float),cudaMemcpyDeviceToHost);
		rsnew=tmpdot_h[0];
		beta=rsnew/rsold;
		rsold=rsnew;      
		mulC_kernel<<<dimGrid_x, dimBlock>>>(p,beta,nx,ny);cudaThreadSynchronize();
		plus_kernel<<<dimGrid_x, dimBlock>>>(p,r,nx,ny);cudaThreadSynchronize();

	}

	cublasDestroy(h);
//	equalC_kernel<<<dimGrid_x, dimBlock>>>(x_d,mu,nx,ny);cudaThreadSynchronize();
//	equal_kernel<<<dimGrid_x, dimBlock>>>(x_d,tmpatax,nx,ny);cudaThreadSynchronize();
//	equalf_kernel_c<<<dimGrid_x, dimBlock>>>(x_d,mask_d,nx,ny);cudaThreadSynchronize();
//	equal_kernel_c<<<dimGrid_x, dimBlock>>>(x_d,mask_d,nx,ny);cudaThreadSynchronize();

}









void ctrecon_conjugate_grad(float *x, float *y, float *xp, float SO,float OD,float scale,int nx,int ny,float *sd_phi,int nd, float *y_det,float dy_det,float y_os,int nt,int *id_X,int *id_Y,int *Nv,int tmp_size,int nv,float *cos_phi,float *sin_phi,float *cos_det,float *sin_det,
  float mu, float rho, int CG_maxit, float CG_tol0, float lambda)
{

 	float *sd_phi_d,*y_det_d,*cos_phi_d,*sin_phi_d,*cos_det_d,*sin_det_d;
	int *id_Y_d,*Nv_d;



	int *id_X_d;



	








	cudaMalloc(&id_X_d,nv*sizeof(int));
	cudaMalloc(&cos_phi_d,nv*sizeof(float));cudaMemcpy(cos_phi_d,cos_phi,nv*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&sin_phi_d,nv*sizeof(float));cudaMemcpy(sin_phi_d,sin_phi,nv*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&cos_det_d,nd*sizeof(float));cudaMemcpy(cos_det_d,cos_det,nd*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&sin_det_d,nd*sizeof(float));cudaMemcpy(sin_det_d,sin_det,nd*sizeof(float),cudaMemcpyHostToDevice);

	cudaMemcpy(id_X_d,id_X,nv*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&sd_phi_d,nv*sizeof(float));cudaMemcpy(sd_phi_d,sd_phi,nv*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&y_det_d,nd*sizeof(float));cudaMemcpy(y_det_d,y_det,nd*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&id_Y_d,nt*tmp_size*sizeof(int));cudaMemcpy(id_Y_d,id_Y,nt*tmp_size*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&Nv_d,nt*sizeof(int));cudaMemcpy(Nv_d,Nv,nt*sizeof(int),cudaMemcpyHostToDevice);





	float *tmpy,  *tmpatax,  *tmpwx,   *tmpwtwx,  *r, *p,  *tmpdot, *b_d;
	float *x_d, *y_d, *xp_d;
	cudaMalloc((void**)&tmpy, sizeof(float)*nv*nd);
	cudaMalloc((void**)&tmpatax, sizeof(float)*nx*ny);
	cudaMalloc((void**)&tmpwx, sizeof(float)*nx*ny*2);
	cudaMalloc((void**)&tmpwtwx, sizeof(float)*nx*ny);
	cudaMalloc((void**)&r, sizeof(float)*nx*ny);
	cudaMalloc((void**)&p, sizeof(float)*nx*ny);
	cudaMalloc(&tmpdot,1*sizeof(float));
	cudaMalloc((void**)&x_d, sizeof(float)*nx*ny);
	cudaMalloc((void**)&y_d, sizeof(float)*nv*nd);
	cudaMalloc((void**)&xp_d, sizeof(float)*nx*ny);
	cudaMalloc((void**)&b_d, sizeof(float)*nx*ny);

	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_x((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	dim3 dimGrid_wx((2*nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);


	cudaMemcpy(x_d, x, sizeof(float)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(float)*nv*nd, cudaMemcpyHostToDevice);
	cudaMemcpy(xp_d, xp, sizeof(float)*nx*ny, cudaMemcpyHostToDevice);

//	cudaMemcpy(x_d, xp, sizeof(float)*nx*ny, cudaMemcpyHostToDevice);


			
//	initialize_kernel_c<<<dimGrid_x, dimBlock>>>(x_dc,nx,ny);

	Atx_fan_mf_gpu_new_kernel<<<dimGrid_x, dimBlock>>>(b_d,y_d,SO,OD,scale,nx,ny,sd_phi_d,nd,y_det_d,dy_det,y_os,nt,id_Y_d,Nv_d,tmp_size,cos_phi_d,sin_phi_d,cos_det_d,sin_det_d);cudaThreadSynchronize();
	wx_2d_d(tmpwx,xp_d,nx,ny);// xp_dc for 
	wtx_2d_d(tmpwtwx,tmpwx,nx,ny);//equal_kernel_c<<<dimGrid_x, dimBlock>>>(x_dc,b_dc,nx,ny);cudaThreadSynchronize();
	mulC_plus_kernel<<<dimGrid_x, dimBlock>>>(b_d,tmpwtwx,mu,nx,ny);cudaThreadSynchronize();
	mulC_plus_kernel<<<dimGrid_x, dimBlock>>>(b_d,xp_d,rho,nx,ny);cudaThreadSynchronize();
	conjugate_grad_d(b_d, x_d, SO, OD, scale, nx, ny, sd_phi_d, nd,  y_det_d, dy_det, y_os, nt, id_X_d, id_Y_d, Nv_d, tmp_size, nv, cos_phi_d, sin_phi_d, cos_det_d, sin_det_d, mu,  rho,  CG_maxit,  CG_tol0,  tmpy,  tmpatax,  tmpwx,  tmpwtwx,  r, p, tmpdot);
//	equalC_kernel<<<dimGrid_x, dimBlock>>>(x_d,mu,nx,ny);cudaThreadSynchronize();
	cudaMemcpy(x, x_d, sizeof(float)*nx*ny, cudaMemcpyDeviceToHost);
  

	cudaFree(tmpy);
	cudaFree(tmpatax);
	cudaFree(tmpwx);
	cudaFree(tmpwtwx);
	cudaFree(r);
	cudaFree(p);
	cudaFree(tmpdot);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(xp_d);
	cudaFree(b_d);
	cudaFree(sd_phi_d);cudaFree(y_det_d);cudaFree(id_Y_d);cudaFree(Nv_d);
   	cudaFree(cos_phi_d);cudaFree(sin_phi_d);cudaFree(cos_det_d);cudaFree(sin_det_d);  cudaFree(id_X_d);

}






void ctrecon_admm_d(float *x_d, float *y_d, float *xp_d, float *b_d, float SO,float OD,float scale,int nx,int ny,float *sd_phi,int nd, float *y_det,float dy_det,float y_os,int nt,int *id_X,int *id_Y,int *Nv,int tmp_size,int nv,float *cos_phi,float *sin_phi,float *cos_det,float *sin_det,
  float mu, float rho, int CG_maxit, int N_iter, float CG_tol0, float lambda,  float *tmpy, float *tmpatax, float *tmpwx,  float *tmpwtwx, float *r,float *p,float *tmpdot, float *d_xs, float *v_xs )
{
	int n_iter;
	float lam=mu/lambda;

		
	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_x((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	dim3 dimGrid_wx((2*nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
		
//	initialize_kernel_c<<<dimGrid_x, dimBlock>>>(x,nx,ny);
//	initialize_kernel_c<<<dimGrid_wx, dimBlock>>>(d_xs,2*nx,ny);
//	initialize_kernel_c<<<dimGrid_wx, dimBlock>>>(v_xs,2*nx,ny);



	    
	for(n_iter=0;n_iter<N_iter;n_iter++)
	{	
		Atx_fan_mf_gpu_new_kernel<<<dimGrid_x, dimBlock>>>(b_d,y_d,SO,OD,scale,nx,ny,sd_phi,nd,y_det,dy_det,y_os,nt,id_Y,Nv,tmp_size,cos_phi,sin_phi,cos_det,sin_det);cudaThreadSynchronize();
		addition_kernel<<<dimGrid_wx, dimBlock>>>(tmpwx,d_xs,v_xs,2*nx,ny);cudaThreadSynchronize();
		wtx_2d_d(tmpwtwx,tmpwx,nx,ny);
//		mulC_kernel<<<dimGrid_x, dimBlock>>>(tmpwtwx,mu,nx,ny);cudaThreadSynchronize();
		mulC_plus_kernel<<<dimGrid_x, dimBlock>>>(b_d,tmpwtwx,mu,nx,ny);cudaThreadSynchronize();
		mulC_plus_kernel<<<dimGrid_x, dimBlock>>>(b_d,xp_d,rho,nx,ny);cudaThreadSynchronize();
		conjugate_grad_d(b_d, x_d, SO, OD, scale, nx, ny, sd_phi, nd, y_det, dy_det, y_os, nt, id_X, id_Y, Nv, tmp_size, nv, cos_phi, sin_phi, cos_det, sin_det, mu,  rho,  CG_maxit,  CG_tol0,  tmpy,  tmpatax,  tmpwx,  tmpwtwx,  r, p, tmpdot);
		wx_2d_d(tmpwx,x_d,nx,ny);
		minus_kernel<<<dimGrid_wx, dimBlock>>>(tmpwx,v_xs,2*nx,ny);cudaThreadSynchronize();
		shrink_2d_d(d_xs,tmpwx,lam,nx,ny);
//		equal_kernel_c<<<dimGrid_wx, dimBlock>>>(d_xs,tmpwx,2*nx,ny);cudaThreadSynchronize();
		subtraction_kernel<<<dimGrid_wx, dimBlock>>>(v_xs,d_xs,tmpwx,2*nx,ny);cudaThreadSynchronize();

	}
//	equalC_kernel<<<dimGrid_x, dimBlock>>>(x_d,float(n_iter),nx,ny);cudaThreadSynchronize();
	
}


void ctrecon_admm(float *x, float *y, float *xp, float SO,float OD,float scale,int nx,int ny,float *sd_phi,int nd, float *y_det,float dy_det,float y_os,int nt,int *id_X, int *id_Y,int *Nv,int tmp_size,int nv,float *cos_phi,float *sin_phi,float *cos_det,float *sin_det,
  float mu, float rho, int CG_maxit, int N_iter, float CG_tol0, float lambda)
  {


 	float *sd_phi_d,*y_det_d,*cos_phi_d,*sin_phi_d,*cos_det_d,*sin_det_d;
	int *id_Y_d, *Nv_d, *id_X_d;
	cudaMalloc(&id_X_d,nv*sizeof(int));
	cudaMalloc(&cos_phi_d,nv*sizeof(float));cudaMemcpy(cos_phi_d,cos_phi,nv*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&sin_phi_d,nv*sizeof(float));cudaMemcpy(sin_phi_d,sin_phi,nv*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&cos_det_d,nd*sizeof(float));cudaMemcpy(cos_det_d,cos_det,nd*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&sin_det_d,nd*sizeof(float));cudaMemcpy(sin_det_d,sin_det,nd*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc(&sd_phi_d,nv*sizeof(float));
	cudaMalloc(&y_det_d,nd*sizeof(float));
	cudaMalloc(&id_Y_d,nt*tmp_size*sizeof(int));

	cudaMemcpy(id_X_d,id_X,nv*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(sd_phi_d,sd_phi,nv*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(y_det_d,y_det,nd*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(id_Y_d,id_Y,nt*tmp_size*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&Nv_d,nt*sizeof(int));cudaMemcpy(Nv_d,Nv,nt*sizeof(int),cudaMemcpyHostToDevice);








	float *tmpy,  *tmpatax,  *tmpwx,   *tmpwtwx,  *r, *p,  *tmpdot, *d_xs, *v_xs, *x_d, *y_d, *xp_d, *b_d;
	cudaMalloc((void**)&tmpy, sizeof(float)*nv*nd);
	cudaMalloc((void**)&tmpatax, sizeof(float)*nx*ny);
	cudaMalloc((void**)&tmpwx, sizeof(float)*nx*ny*2);
	cudaMalloc((void**)&tmpwtwx, sizeof(float)*nx*ny);
	cudaMalloc((void**)&r, sizeof(float)*nx*ny);
	cudaMalloc((void**)&p, sizeof(float)*nx*ny);
	cudaMalloc(&tmpdot,1*sizeof(float));
	cudaMalloc((void**)&x_d, sizeof(float)*nx*ny);
	cudaMalloc((void**)&y_d, sizeof(float)*nv*nd);
	cudaMalloc((void**)&xp_d, sizeof(float)*nx*ny);
	cudaMalloc((void**)&b_d, sizeof(float)*nx*ny);
	cudaMalloc((void**)&d_xs, sizeof(float)*nx*ny*2);
	cudaMalloc((void**)&v_xs, sizeof(float)*nx*ny*2);

	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_x((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	dim3 dimGrid_wx((2*nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	cudaMemcpy(x_d, x, sizeof(float)*nx*ny, cudaMemcpyHostToDevice);
	cudaMemcpy(y_d, y, sizeof(float)*nv*nd, cudaMemcpyHostToDevice);
	cudaMemcpy(xp_d, xp, sizeof(float)*nx*ny, cudaMemcpyHostToDevice);	

	initialize_kernel<<<dimGrid_wx, dimBlock>>>(d_xs,2*nx,ny);
	initialize_kernel<<<dimGrid_wx, dimBlock>>>(v_xs,2*nx,ny);

	ctrecon_admm_d(x_d, y_d, xp_d, b_d, SO, OD, scale, nx, ny, sd_phi_d, nd,  y_det_d, dy_det, y_os, nt, id_X_d, id_Y_d, Nv_d, tmp_size, nv, cos_phi_d, sin_phi_d, cos_det_d, sin_det_d, mu,  rho,  CG_maxit, N_iter, CG_tol0, lambda, tmpy,  tmpatax,  tmpwx,  tmpwtwx,  r, p, tmpdot,d_xs,v_xs);


	cudaMemcpy(x, x_d, sizeof(float)*nx*ny, cudaMemcpyDeviceToHost);

   
	cudaFree(tmpy);
	cudaFree(tmpatax);
	cudaFree(tmpwx);
	cudaFree(tmpwtwx);
	cudaFree(r);
	cudaFree(p);
	cudaFree(tmpdot);
	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(xp_d);
	cudaFree(b_d);
	cudaFree(d_xs);
	cudaFree(v_xs);

	cudaFree(sd_phi_d);cudaFree(y_det_d);cudaFree(id_Y_d);cudaFree(Nv_d);
   	cudaFree(cos_phi_d);cudaFree(sin_phi_d);cudaFree(cos_det_d);cudaFree(sin_det_d);  cudaFree(id_X_d);



}




