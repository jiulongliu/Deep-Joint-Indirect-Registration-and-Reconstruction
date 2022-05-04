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
#include <cufft.h>
#include <cublas.h>
#include <cublas_v2.h>
//#include <cufftXt.h>
#define ABS(a) (a>0?a:-(a))

#define BLOCK_SIZE_x 16
#define BLOCK_SIZE_y 16

extern "C" void fft2d(float *ur, float *ui, float *fr, float *fi, int N);
extern "C" void ifft2d(float *ur, float *ui, float *fr, float *fi, int N);
extern "C" void Ax_mri(float *ur, float *ui, float *fr, float *fi, float *mask,int N );
extern "C" void Atx_mri(float *ur, float *ui, float *fr, float *fi, float *mask, int N );
extern "C" void conjugate_grad(float *br, float *bi, float *xr, float *xi,  float mu, float rho, int maxit, float CG_tol0, float *mask, int N );
extern "C" void mrirecon_conjugate_grad(float *xr, float *yr, float *yi, float *xpr,  float mu, float rho, int CG_maxit, float CG_tol0, float *mask,int N, float lambda);
extern "C" void mrirecon_admm(float *xr, float *yr, float *yi, float *xpr,  float mu, float rho, int CG_maxit, float CG_tol0, int N_iter, float *mask,int N, float lambda);

__global__ void real2complex(float *fr, float *fi, cufftComplex *fc, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j*N + i;
	if (i<N && j<N)
	{
		fc[index].x = fr[index];
		fc[index].y = fi[index];
	}
}

__global__ void complex2real(cufftComplex *fc, float *fr, float *fi,int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j*N + i;
	if (i<N && j<N)
	{
		fr[index] = fc[index].x;
		fi[index] = fc[index].y;
	}
}

__global__ void complex2real_scale(cufftComplex *fc, float *fr, float *fi,int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j*N + i;
	if (i<N && j<N)
	{
		fr[index] = fc[index].x/(float)N;//((float)N*(float)N);
		fi[index] = fc[index].y/(float)N;//((float)N*(float)N);
		//divide by number of elements to recover value
	}
}


__global__ void complex2magtitude(cufftComplex *fc, float *fr, int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j*N + i;
	if (i<N && j<N)
	{
		fr[index] = sqrt(fc[index].x*fc[index].x+fc[index].y*fc[index].y);
		
	}
}



__global__ void kspacedownsample(cufftComplex *fc, float *mask,int N)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j*N + i;
	if (i<N && j<N)
	{
		fc[index].x = fc[index].x*mask[index];
		fc[index].y = fc[index].y*mask[index];
		
	}
}




void fft2d(float *ur, float *ui, float *fr, float *fi, int N )
{   


//	N=192;
	float  *fr_d, *fi_d, *ur_d, *ui_d;//*k_d,

	cudaMalloc((void**)&fr_d, sizeof(float)*N*N);
	cudaMalloc((void**)&fi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&ur_d, sizeof(float)*N*N);
	cudaMalloc((void**)&ui_d, sizeof(float)*N*N);

	cudaMemcpy(fr_d, fr, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(fi_d, fi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cufftComplex *f_fft_dc, *f_dc;
	cudaMalloc((void**)&f_fft_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&f_dc, sizeof(cufftComplex)*N*N);
	dim3 dimBlock (BLOCK_SIZE_x, BLOCK_SIZE_y);
	dim3 dimGrid_t((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y);
	real2complex<<<dimGrid_t, dimBlock>>>(fr_d, fi_d, f_dc, N);
	cufftHandle plan;
	cufftPlan2d(&plan, N, N, CUFFT_C2C);

	cufftExecC2C(plan, f_dc, f_fft_dc , CUFFT_FORWARD);
//	cufftExecC2C(plan, f_fft_dc, f_dc, CUFFT_INVERSE);
	complex2real_scale<<<dimGrid_t, dimBlock>>>(f_fft_dc, ur_d, ui_d, N);
//	complex2real<<<dimGrid_t, dimBlock>>>(f_dc, ur_d, ui_d, N);
	cudaMemcpy(ur, ur_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(ui, ui_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

	cufftDestroy(plan);
	cudaFree(fr_d);
	cudaFree(fi_d);
	cudaFree(ur_d);
	cudaFree(ui_d);
	cudaFree(f_fft_dc);
	cudaFree(f_dc);
}



void ifft2d(float *ur, float *ui, float *fr, float *fi, int N )
{   


//	N=192;
	float  *fr_d, *fi_d, *ur_d, *ui_d;//*k_d,

	cudaMalloc((void**)&fr_d, sizeof(float)*N*N);
	cudaMalloc((void**)&fi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&ur_d, sizeof(float)*N*N);
	cudaMalloc((void**)&ui_d, sizeof(float)*N*N);

	cudaMemcpy(fr_d, fr, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(fi_d, fi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cufftComplex *f_ifft_dc, *f_dc;
	cudaMalloc((void**)&f_ifft_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&f_dc, sizeof(cufftComplex)*N*N);
	dim3 dimBlock (BLOCK_SIZE_x, BLOCK_SIZE_y);
	dim3 dimGrid_t((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y);
	real2complex<<<dimGrid_t, dimBlock>>>(fr_d, fi_d, f_dc, N);
	cufftHandle plan;
	cufftPlan2d(&plan, N, N, CUFFT_C2C);

//	cufftExecC2C(plan, f_dc, f_ifft_dc , CUFFT_FORWARD);
	cufftExecC2C(plan, f_dc, f_ifft_dc, CUFFT_INVERSE);
	complex2real_scale<<<dimGrid_t, dimBlock>>>(f_ifft_dc, ur_d, ui_d, N);
	cudaMemcpy(ur, ur_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(ui, ui_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	cudaFree(fr_d);
	cudaFree(fi_d);
	cudaFree(ur_d);
	cudaFree(ui_d);
	cudaFree(f_ifft_dc);
	cudaFree(f_dc);

}


void Ax_mri(float *ur, float *ui, float *fr, float *fi, float *mask,int N )
{   


//	N=192;
	float  *fr_d, *fi_d, *ur_d, *ui_d, *mask_d;//*k_d,

	cudaMalloc((void**)&fr_d, sizeof(float)*N*N);
	cudaMalloc((void**)&fi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&ur_d, sizeof(float)*N*N);
	cudaMalloc((void**)&ui_d, sizeof(float)*N*N);
	cudaMalloc((void**)&mask_d, sizeof(float)*N*N);
	cudaMemcpy(fr_d, fr, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(fi_d, fi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(mask_d, mask, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cufftComplex *f_fft_dc, *f_dc;
	cudaMalloc((void**)&f_fft_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&f_dc, sizeof(cufftComplex)*N*N);
	dim3 dimBlock (BLOCK_SIZE_x, BLOCK_SIZE_y);
	dim3 dimGrid_t((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y);
	real2complex<<<dimGrid_t, dimBlock>>>(fr_d, fi_d, f_dc, N);
	cufftHandle plan;
	cufftPlan2d(&plan, N, N, CUFFT_C2C);

	cufftExecC2C(plan, f_dc, f_fft_dc , CUFFT_FORWARD);
	kspacedownsample<<<dimGrid_t, dimBlock>>>(f_fft_dc, mask_d, N);
	complex2real_scale<<<dimGrid_t, dimBlock>>>(f_fft_dc, ur_d, ui_d, N);

	cudaMemcpy(ur, ur_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(ui, ui_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	cudaFree(fr_d);
	cudaFree(fi_d);
	cudaFree(ur_d);
	cudaFree(ui_d);
	cudaFree(f_fft_dc);
	cudaFree(f_dc);
}


void Atx_mri(float *ur, float *ui, float *fr, float *fi, float *mask, int N )
{   


//	N=192;
	float  *fr_d, *fi_d, *ur_d, *ui_d, *mask_d;//*k_d,

	cudaMalloc((void**)&fr_d, sizeof(float)*N*N);
	cudaMalloc((void**)&fi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&ur_d, sizeof(float)*N*N);
	cudaMalloc((void**)&ui_d, sizeof(float)*N*N);
	cudaMalloc((void**)&mask_d, sizeof(float)*N*N);
	cudaMemcpy(fr_d, fr, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(fi_d, fi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(mask_d, mask, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cufftComplex *f_ifft_dc, *f_dc;
	cudaMalloc((void**)&f_ifft_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&f_dc, sizeof(cufftComplex)*N*N);
	dim3 dimBlock (BLOCK_SIZE_x, BLOCK_SIZE_y);
	dim3 dimGrid_t((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y);
	real2complex<<<dimGrid_t, dimBlock>>>(fr_d, fi_d, f_dc, N);
	kspacedownsample<<<dimGrid_t, dimBlock>>>(f_dc, mask_d, N);
	cufftHandle plan;
	cufftPlan2d(&plan, N, N, CUFFT_C2C);

//	cufftExecC2C(plan, f_dc, f_ifft_dc , CUFFT_FORWARD);
	cufftExecC2C(plan, f_dc, f_ifft_dc, CUFFT_INVERSE);
	complex2real_scale<<<dimGrid_t, dimBlock>>>(f_ifft_dc, ur_d, ui_d, N);
	cudaMemcpy(ur, ur_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(ui, ui_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	cudaFree(fr_d);
	cudaFree(fi_d);
	cudaFree(ur_d);
	cudaFree(ui_d);
	cudaFree(f_ifft_dc);
	cudaFree(f_dc);
}







__global__ void wx_2d_kernel_c(cufftComplex *wx, cufftComplex *x,int nx,int ny)
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
			wx[2 * idx].x = x[idx].x - x[idx + 1].x; // check
			wx[2 * idx].y = x[idx].y - x[idx + 1].y;
		}
		else
		{
			wx[2 * idx].x = 0.0f;
			wx[2 * idx].y = 0.0f;
		}
		if(iy<ny-1)		
		{

			wx[2 * idx + 1].x = x[idx].x - x[idx + nx].x;
			wx[2 * idx + 1].y = x[idx].y - x[idx + nx].y;
		}
		else
		{
			wx[2 * idx + 1].x = 0.0f;
			wx[2 * idx + 1].y = 0.0f;
		}
	}
}
void wx_2d_dc(cufftComplex *wx_d, cufftComplex *x_d,int nx,int ny)
{   dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_t((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	wx_2d_kernel_c<<<dimGrid_t, dimBlock>>>(wx_d,x_d,nx,ny);
	cudaThreadSynchronize();			
}

__global__ void wtx_2d_kernel_c(cufftComplex *x, cufftComplex *wx,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	
		int idx;
		idx=iy*nx+ix;
		x[idx].x = 0.0f;
		x[idx].y = 0.0f;
		if(ix==0)		
		{
			x[idx].x+=wx[2*idx].x;
			x[idx].y+=wx[2*idx].y;

		}
		else
		{	if(ix<nx-1)
			{
				x[idx].x+=-wx[2*(idx-1)].x+wx[2*idx].x;
				x[idx].y+=-wx[2*(idx-1)].y+wx[2*idx].y;
			}
			else
			{
				x[idx].x+=-wx[2*(idx-1)].x;
				x[idx].y+=-wx[2*(idx-1)].y;
			}
		}
		if(iy==0)		
		{
			x[idx].x+=wx[2*idx+1].x;
			x[idx].y+=wx[2*idx+1].y;
		}
		else
		{	if(iy<ny-1)
			{
				x[idx].x+=-wx[2*(idx-nx)+1].x+wx[2*idx+1].x;
				x[idx].y+=-wx[2*(idx-nx)+1].y+wx[2*idx+1].y;
			}
			else
			{
				x[idx].x+=-wx[2*(idx-nx)+1].x;
				x[idx].y+=-wx[2*(idx-nx)+1].y;
			}
		}
		
	}
}
void wtx_2d_dc(cufftComplex *x_d, cufftComplex *wx_d,int nx,int ny)
{   dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_t((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);			        
	wtx_2d_kernel_c<<<dimGrid_t, dimBlock>>>(x_d,wx_d,nx,ny);
	cudaThreadSynchronize();			
}

__global__ void shrink_2d_kernel_c(cufftComplex *y, cufftComplex *x,float s,int nx,int ny)
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
		tmp=(float)sqrt(x[idx].x*x[idx].x-x[idx].y*x[idx].y+x[idx+1].x*x[idx+1].x-x[idx+1].y*x[idx+1].y);
        if(tmp>s)
        {
			d=(tmp-s)/tmp;
			y[idx].x=x[idx].x*d;
			y[idx].y=x[idx].y*d;
			y[idx+1].x=x[idx+1].x*d;
			y[idx+1].y=x[idx+1].y*d;
		}
        else
        {
			y[idx].x=0.0f;
			y[idx].y=0.0f;
			y[idx+1].x=0.0f;
			y[idx+1].y=0.0f;
		}
	}
}
void shrink_2d_dc(cufftComplex *y, cufftComplex *x,float s,int nx,int ny)
{   
	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_t((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	shrink_2d_kernel_c<<<dimGrid_t, dimBlock>>>(y,x,s,nx,ny);
	cudaThreadSynchronize();			
}

__global__ void mulC_kernel_c(cufftComplex *x, float c,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		x[idx].x *= c;
		x[idx].y *= c;
	}
}
__global__ void mulC_plus_kernel_c(cufftComplex *y, cufftComplex *x,float s,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		y[idx].x += s*x[idx].x;
		y[idx].y += s*x[idx].y;
	}
}

__global__ void plus_kernel_c(cufftComplex *y, cufftComplex *x,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		y[idx].x += x[idx].x;
		y[idx].y += x[idx].y;
	}
}

__global__ void addition_kernel_c(cufftComplex *z,cufftComplex *y, cufftComplex *x,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		z[idx].x = y[idx].x + x[idx].x;
		z[idx].y = y[idx].y + x[idx].y;
	}
}


__global__ void subtraction_kernel_c(cufftComplex *z,cufftComplex *y, cufftComplex *x,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		z[idx].x = y[idx].x - x[idx].x;
		z[idx].y = y[idx].y - x[idx].y;
	}
}


__global__ void minus_kernel_c(cufftComplex *y, cufftComplex *x,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		y[idx].x -= x[idx].x;
		y[idx].y -= x[idx].y;
	}
}


__global__ void initialize_kernel_c(cufftComplex *y,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		y[idx].x = 0.0f;
		y[idx].y = 0.0f;
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
__global__ void equal_kernel_c(cufftComplex *y, cufftComplex *x,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		y[idx].x = x[idx].x;
		y[idx].y = x[idx].y;
	}
}


__global__ void equalC_kernel_c(cufftComplex *y, float c,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		y[idx].x = c;
		y[idx].y = c+0.10f;
	}
}
__global__ void equalf_kernel_c(cufftComplex *y, float *c,int nx,int ny)
{	int bx=blockIdx.x;
	int by=blockIdx.y;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int ix=bx*BLOCK_SIZE_x+tx;
	int iy=by*BLOCK_SIZE_y+ty;
    if(ix<nx&&iy<ny)
	{	int idx=(iy*nx+ix);		
		y[idx].x = c[idx];
		y[idx].y = c[idx]+0.10f;
	}
}
void Ax_mri_d(cufftComplex *fc,cufftComplex *uc, float *mask_d,int N, cufftHandle plan)
{   



	dim3 dimBlock (BLOCK_SIZE_x, BLOCK_SIZE_y);
	dim3 dimGrid_t((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y);
//	cufftHandle plan;
//	cufftPlan2d(&plan, N, N, CUFFT_C2C);

	cufftExecC2C(plan, uc, fc , CUFFT_FORWARD);
	mulC_kernel_c<<<dimGrid_t, dimBlock>>>(fc, 1.0f/((float)N), N, N);
	kspacedownsample<<<dimGrid_t, dimBlock>>>(fc, mask_d, N);
//	cufftDestroy(plan);
}



void Atx_mri_d(cufftComplex *uc,cufftComplex *fc, float *mask_d,int N, cufftHandle plan)
{   

	dim3 dimBlock (BLOCK_SIZE_x, BLOCK_SIZE_y);
	dim3 dimGrid_t((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y);

//	cufftHandle plan;
//	cufftPlan2d(&plan, N, N, CUFFT_C2C);

	kspacedownsample<<<dimGrid_t, dimBlock>>>(fc, mask_d, N);
	cufftExecC2C(plan, fc, uc, CUFFT_INVERSE);
	mulC_kernel_c<<<dimGrid_t, dimBlock>>>(uc, 1.0f/((float)N), N, N);
//	cufftDestroy(plan);
}

void conjugate_grad_dc(cufftComplex *b_d, cufftComplex *x_d,  float mu, float rho, int CG_maxit, float CG_tol0, float *mask_d,int N, cufftComplex *tmpy, cufftComplex *tmpatax, cufftComplex *tmpwx,  cufftComplex *tmpwtwx, cufftComplex *r,cufftComplex *p,cufftComplex *tmpdot, float *tmpdotr, float *tmpdoti, cufftHandle plan)
{

	int nx=N, ny=N, iter;	
	float rsold,rsnew,tmpdot_h[1],CG_tol,alpha,beta;
	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_x((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);


	Ax_mri_d(tmpy,x_d, mask_d, N, plan);
	Atx_mri_d(tmpatax,tmpy, mask_d, N, plan);
	wx_2d_dc(tmpwx,x_d,nx,ny);
	wtx_2d_dc(tmpwtwx,tmpwx,nx,ny);	
	mulC_plus_kernel_c<<<dimGrid_x, dimBlock>>>(tmpatax,tmpwtwx,mu,nx,ny);cudaThreadSynchronize();
	mulC_plus_kernel_c<<<dimGrid_x, dimBlock>>>(tmpatax,x_d,rho,nx,ny);cudaThreadSynchronize();
	subtraction_kernel_c<<<dimGrid_x, dimBlock>>>(r,b_d,tmpatax,nx,ny);cudaThreadSynchronize();
	equal_kernel_c<<<dimGrid_x, dimBlock>>>(p,r,nx,ny);cudaThreadSynchronize();
	cublasHandle_t h;
	cublasCreate(&h);
	cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
//	equalC_kernel_c<<<dimGrid_x, dimBlock>>>(r,(float)0.3,nx,ny);cudaThreadSynchronize(); //will be 0.25 * number of pixels
	cublasCdotc(h,nx*ny,r,1,r,1,tmpdot);//tmpdot tmpdot_h2
	complex2real<<<dimGrid_x, dimBlock>>>(tmpdot, tmpdotr, tmpdoti, 1);
	cudaMemcpy(tmpdot_h,tmpdotr,1*sizeof(float),cudaMemcpyDeviceToHost);
	rsold=tmpdot_h[0];
	CG_tol=rsold*CG_tol0;
	
	for(iter=0; iter<CG_maxit && rsold>CG_tol;iter++) //&& rsold>CG_tol CG_maxit
	{
		Ax_mri_d(tmpy,p, mask_d, N, plan);
		Atx_mri_d(tmpatax,tmpy, mask_d, N, plan);
		wx_2d_dc(tmpwx,p,nx,ny);
		wtx_2d_dc(tmpwtwx,tmpwx,nx,ny);	

		mulC_plus_kernel_c<<<dimGrid_x, dimBlock>>>(tmpatax,tmpwtwx,mu,nx,ny);cudaThreadSynchronize();
		mulC_plus_kernel_c<<<dimGrid_x, dimBlock>>>(tmpatax,p,rho,nx,ny);cudaThreadSynchronize();
		cublasCdotc(h,nx*ny,p,1,tmpatax,1,tmpdot);
		complex2real<<<dimGrid_x, dimBlock>>>(tmpdot, tmpdotr, tmpdoti, 1);
		cudaMemcpy(tmpdot_h,tmpdotr,1*sizeof(float),cudaMemcpyDeviceToHost);
		rsnew=tmpdot_h[0];
		if(rsnew<CG_tol) break;
		alpha=rsold/rsnew;
		mulC_plus_kernel_c<<<dimGrid_x, dimBlock>>>(x_d,p,alpha,nx,ny);cudaThreadSynchronize();
		mulC_plus_kernel_c<<<dimGrid_x, dimBlock>>>(r,tmpatax,-alpha,nx,ny);cudaThreadSynchronize();
		cublasCdotc(h,nx*ny,r,1,r,1,tmpdot);
		complex2real<<<dimGrid_x, dimBlock>>>(tmpdot, tmpdotr, tmpdoti, 1);
		cudaMemcpy(tmpdot_h,tmpdotr,1*sizeof(float),cudaMemcpyDeviceToHost);
		rsnew=tmpdot_h[0];
		beta=rsnew/rsold;
		rsold=rsnew;      
		mulC_kernel_c<<<dimGrid_x, dimBlock>>>(p,beta,nx,ny);cudaThreadSynchronize();
		plus_kernel_c<<<dimGrid_x, dimBlock>>>(p,r,nx,ny);cudaThreadSynchronize();

	}

	cublasDestroy(h);
//	equalC_kernel_c<<<dimGrid_x, dimBlock>>>(x_d,3.0,nx,ny);cudaThreadSynchronize();
//	equalf_kernel_c<<<dimGrid_x, dimBlock>>>(x_d,mask_d,nx,ny);cudaThreadSynchronize();
//	equal_kernel_c<<<dimGrid_x, dimBlock>>>(x_d,mask_d,nx,ny);cudaThreadSynchronize();

}

void conjugate_grad(float *br, float *bi, float *xr, float *xi,  float mu, float rho, int CG_maxit, float CG_tol0, float *mask, int N)
{

	
	//	N=192;
	float  *br_d, *bi_d, *xr_d, *xi_d, *mask_d;//*k_d,
	float *tmpdotr, *tmpdoti;
	cufftComplex *tmpy, *tmpatax, *tmpwx,  *tmpwtwx, *r, *p, *tmpdot;
	cufftHandle plan;
	cufftPlan2d(&plan, N, N, CUFFT_C2C);
	cudaMalloc((void**)&tmpy, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&tmpatax, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&tmpwx, sizeof(cufftComplex)*N*N*2);
	cudaMalloc((void**)&tmpwtwx, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&r, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&p, sizeof(cufftComplex)*N*N);
	cudaMalloc(&tmpdot,1*sizeof(cufftComplex));
	cudaMalloc(&tmpdotr,1*sizeof(float));
	cudaMalloc(&tmpdoti,1*sizeof(float));

	cudaMalloc((void**)&br_d, sizeof(float)*N*N);
	cudaMalloc((void**)&bi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&xr_d, sizeof(float)*N*N);
	cudaMalloc((void**)&xi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&mask_d, sizeof(float)*N*N);
	cudaMemcpy(br_d, br, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(bi_d, bi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(xr_d, xr, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(xi_d, xi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(mask_d, mask, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cufftComplex *b_dc, *x_dc;
	cudaMalloc((void**)&b_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&x_dc, sizeof(cufftComplex)*N*N);

	dim3 dimBlock (BLOCK_SIZE_x, BLOCK_SIZE_y);
	dim3 dimGrid_x((N+dimBlock.x-1)/dimBlock.x,(N+dimBlock.y-1)/dimBlock.y);
	real2complex<<<dimGrid_x, dimBlock>>>(br_d, bi_d, b_dc, N);
	real2complex<<<dimGrid_x, dimBlock>>>(xr_d, xi_d, x_dc, N);

	conjugate_grad_dc(b_dc, x_dc, mu, rho,  CG_maxit, CG_tol0, mask_d, N, tmpy, tmpatax, tmpwx,  tmpwtwx, r, p, tmpdot, tmpdotr, tmpdoti, plan);

	complex2real<<<dimGrid_x, dimBlock>>>(x_dc, xr_d, xi_d, N);

	cudaMemcpy(xr, xr_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(xi, xi_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
	cufftDestroy(plan);
	cudaFree(b_dc);
	cudaFree(tmpy);
	cudaFree(tmpatax);
	cudaFree(tmpwx);
	cudaFree(tmpwtwx);
	cudaFree(r);
	cudaFree(p);
	cudaFree(tmpdot);
	cudaFree(tmpdotr);
	cudaFree(tmpdoti);

}







void mrirecon_conjugate_grad(float *xr, float *yr, float *yi, float *xpr,   float mu, float rho, int CG_maxit, float CG_tol0, float *mask,int N, float lambda)
{

	int nx=N,ny=N;



	cufftComplex *tmpy,  *tmpatax,  *tmpwx,   *tmpwtwx,  *r, *p,  *tmpdot,  *x_dc, *y_dc, *xp_dc, *b_dc;
	float *tmpdotr,  *tmpdoti, *mask_d; 
	float *xr_d, *xi_d, *yr_d, *yi_d, *xpr_d, *xpi_d;
	cudaMalloc((void**)&tmpy, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&tmpatax, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&tmpwx, sizeof(cufftComplex)*N*N*2);
	cudaMalloc((void**)&tmpwtwx, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&r, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&p, sizeof(cufftComplex)*N*N);
	cudaMalloc(&tmpdot,1*sizeof(cufftComplex));
	cudaMalloc(&tmpdotr,1*sizeof(float));
	cudaMalloc(&tmpdoti,1*sizeof(float));
	cudaMalloc((void**)&xr_d, sizeof(float)*N*N);
	cudaMalloc((void**)&xi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&yr_d, sizeof(float)*N*N);
	cudaMalloc((void**)&yi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&xpr_d, sizeof(float)*N*N);
	cudaMalloc((void**)&xpi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&mask_d, sizeof(float)*N*N);
	cudaMalloc((void**)&x_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&y_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&xp_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&b_dc, sizeof(cufftComplex)*N*N);

	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_x((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	dim3 dimGrid_wx((2*nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	cufftHandle plan;
	cufftPlan2d(&plan, N, N, CUFFT_C2C);

	cudaMemcpy(xr_d, xr, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	initialize_kernel<<<dimGrid_x, dimBlock>>>(xi_d,nx,ny);
//	cudaMemcpy(xi_d, xi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(yr_d, yr, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(yi_d, yi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(xpr_d, xpr, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	initialize_kernel<<<dimGrid_x, dimBlock>>>(xpi_d,nx,ny);
//	cudaMemcpy(xpi_d, xpi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(mask_d, mask, sizeof(float)*N*N, cudaMemcpyHostToDevice);


	real2complex<<<dimGrid_x, dimBlock>>>(xr_d, xi_d, x_dc, N);
	real2complex<<<dimGrid_x, dimBlock>>>(yr_d, yi_d, y_dc, N);
	real2complex<<<dimGrid_x, dimBlock>>>(xpr_d, xpi_d, xp_dc, N);
			
//	initialize_kernel_c<<<dimGrid_x, dimBlock>>>(x_dc,nx,ny);
	Atx_mri_d(b_dc, y_dc, mask_d, N, plan);
	wx_2d_dc(tmpwx,x_dc,nx,ny);// xp_dc for 
	wtx_2d_dc(tmpwtwx,tmpwx,nx,ny);//equal_kernel_c<<<dimGrid_x, dimBlock>>>(x_dc,b_dc,nx,ny);cudaThreadSynchronize();
	mulC_plus_kernel_c<<<dimGrid_x, dimBlock>>>(b_dc,tmpwtwx,mu,nx,ny);cudaThreadSynchronize();
	mulC_plus_kernel_c<<<dimGrid_x, dimBlock>>>(b_dc,xp_dc,rho,nx,ny);cudaThreadSynchronize();
	conjugate_grad_dc(b_dc, x_dc, mu, rho,  CG_maxit, CG_tol0, mask_d, N, tmpy, tmpatax, tmpwx,  tmpwtwx, r, p, tmpdot, tmpdotr, tmpdoti, plan);
	complex2magtitude<<<dimGrid_x, dimBlock>>>(x_dc, xr_d, N);	
// 	complex2real<<<dimGrid_x, dimBlock>>>(x_dc, xr_d, xi_d, N);

	cudaMemcpy(xr, xr_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
//	cudaMemcpy(xi, xi_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
  
	cufftDestroy(plan);
	cudaFree(tmpy);
	cudaFree(tmpatax);
	cudaFree(tmpwx);
	cudaFree(tmpwtwx);
	cudaFree(r);
	cudaFree(p);
	cudaFree(tmpdot);
	cudaFree(tmpdotr);
	cudaFree(tmpdoti);
	cudaFree(xr_d);
	cudaFree(xi_d);
	cudaFree(yr_d);
	cudaFree(yi_d);
	cudaFree(xpr_d);
	cudaFree(xpi_d);
	cudaFree(mask_d);
	cudaFree(x_dc);
	cudaFree(y_dc);
	cudaFree(xp_dc);
	cudaFree(b_dc);

}






void mrirecon_admm_dc(cufftComplex *x_d, cufftComplex *y_d, cufftComplex *xp_d, cufftComplex *b_d,  float mu, float rho, int CG_maxit, float CG_tol0, int N_iter, float *mask_d,int N, float lambda, cufftComplex *tmpy, cufftComplex *tmpatax, cufftComplex *tmpwx,  cufftComplex *tmpwtwx, cufftComplex *r,cufftComplex *p, cufftComplex *tmpdot, float *tmpdotr, float *tmpdoti, cufftComplex *d_xs, cufftComplex *v_xs)
{
	int nx=N,ny=N,n_iter;
	float lam=mu/lambda;

	cufftHandle plan;
	cufftPlan2d(&plan, N, N, CUFFT_C2C);


		
	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_x((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	dim3 dimGrid_wx((2*nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
		
//	initialize_kernel_c<<<dimGrid_x, dimBlock>>>(x,nx,ny);
//	initialize_kernel_c<<<dimGrid_wx, dimBlock>>>(d_xs,2*nx,ny);
//	initialize_kernel_c<<<dimGrid_wx, dimBlock>>>(v_xs,2*nx,ny);



	    
	for(n_iter=0;n_iter<N_iter;n_iter++)
	{	
		Atx_mri_d(b_d, y_d, mask_d, N,plan);
		addition_kernel_c<<<dimGrid_wx, dimBlock>>>(tmpwx,d_xs,v_xs,2*nx,ny);cudaThreadSynchronize();
		wtx_2d_dc(tmpwtwx,tmpwx,nx,ny);
//		mulC_kernel<<<dimGrid_x, dimBlock>>>(tmpwtwx,mu,nx,ny);cudaThreadSynchronize();
		mulC_plus_kernel_c<<<dimGrid_x, dimBlock>>>(b_d,tmpwtwx,mu,nx,ny);cudaThreadSynchronize();
		mulC_plus_kernel_c<<<dimGrid_x, dimBlock>>>(b_d,xp_d,rho,nx,ny);cudaThreadSynchronize();
		conjugate_grad_dc(b_d, x_d, mu, rho,  CG_maxit, CG_tol0, mask_d, N, tmpy, tmpatax, tmpwx,  tmpwtwx, r, p, tmpdot, tmpdotr, tmpdoti,plan);
	    //conjugate_grad_dc(b_dc, x_dc, mu, rho,  CG_maxit, CG_tol0, mask_d, N, tmpy, tmpatax, tmpwx,  tmpwtwx, r, p, tmpdot, tmpdotr, tmpdoti);
		wx_2d_dc(tmpwx,x_d,nx,ny);
		minus_kernel_c<<<dimGrid_wx, dimBlock>>>(tmpwx,v_xs,2*nx,ny);cudaThreadSynchronize();
		shrink_2d_dc(d_xs,tmpwx,lam,nx,ny);
//		equal_kernel_c<<<dimGrid_wx, dimBlock>>>(d_xs,tmpwx,2*nx,ny);cudaThreadSynchronize();
		subtraction_kernel_c<<<dimGrid_wx, dimBlock>>>(v_xs,d_xs,tmpwx,2*nx,ny);cudaThreadSynchronize();

	}

	cufftDestroy(plan);
//	initialize_kernel_c<<<dimGrid_x, dimBlock>>>(x_d,nx,ny);
	
}


void mrirecon_admm(float *xr, float *yr, float *yi, float *xpr,  float mu, float rho, int CG_maxit, float CG_tol0, int N_iter, float *mask,int N, float lambda)
{
//	float *tmpy,tmpdot_h[1],s,s2,lam,alpha,beta,CG_tol;
//	int n_iter,n_cg;
	int nx=N,ny=N;



	cufftComplex *tmpy,  *tmpatax,  *tmpwx,   *tmpwtwx,  *r, *p,  *tmpdot, *d_xs, *v_xs, *x_dc, *y_dc, *xp_dc, *b_dc;
	float *tmpdotr,  *tmpdoti; 
	float *xr_d, *xi_d, *yr_d, *yi_d, *xpr_d, *xpi_d, *mask_d;
	cudaMalloc((void**)&tmpy, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&tmpatax, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&tmpwx, sizeof(cufftComplex)*N*N*2);
	cudaMalloc((void**)&tmpwtwx, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&r, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&p, sizeof(cufftComplex)*N*N);
	cudaMalloc(&tmpdot,1*sizeof(cufftComplex));
	cudaMalloc(&tmpdotr,1*sizeof(float));
	cudaMalloc(&tmpdoti,1*sizeof(float));
	cudaMalloc((void**)&xr_d, sizeof(float)*N*N);
	cudaMalloc((void**)&xi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&yr_d, sizeof(float)*N*N);
	cudaMalloc((void**)&yi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&xpr_d, sizeof(float)*N*N);
	cudaMalloc((void**)&xpi_d, sizeof(float)*N*N);
	cudaMalloc((void**)&mask_d, sizeof(float)*N*N);
	cudaMalloc((void**)&x_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&y_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&xp_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&b_dc, sizeof(cufftComplex)*N*N);
	cudaMalloc((void**)&d_xs, sizeof(cufftComplex)*N*N*2);
	cudaMalloc((void**)&v_xs, sizeof(cufftComplex)*N*N*2);

	dim3 dimBlock(BLOCK_SIZE_x,BLOCK_SIZE_y);
	dim3 dimGrid_x((nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	dim3 dimGrid_wx((2*nx+dimBlock.x-1)/dimBlock.x,(ny+dimBlock.y-1)/dimBlock.y);
	cudaMemcpy(xr_d, xr, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	initialize_kernel<<<dimGrid_x, dimBlock>>>(xi_d,nx,ny);
//	cudaMemcpy(xi_d, xi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(yr_d, yr, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(yi_d, yi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(xpr_d, xpr, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	initialize_kernel<<<dimGrid_x, dimBlock>>>(xpi_d,nx,ny);
//	cudaMemcpy(xpi_d, xpi, sizeof(float)*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(mask_d, mask, sizeof(float)*N*N, cudaMemcpyHostToDevice);


	real2complex<<<dimGrid_x, dimBlock>>>(xr_d, xi_d, x_dc, N);
	real2complex<<<dimGrid_x, dimBlock>>>(yr_d, yi_d, y_dc, N);
	real2complex<<<dimGrid_x, dimBlock>>>(xpr_d, xpi_d, xp_dc, N);


	

		

	initialize_kernel_c<<<dimGrid_wx, dimBlock>>>(d_xs,2*nx,ny);
	initialize_kernel_c<<<dimGrid_wx, dimBlock>>>(v_xs,2*nx,ny);

	mrirecon_admm_dc(x_dc, y_dc, xp_dc, b_dc, mu, rho, CG_maxit, CG_tol0, N_iter, mask_d, N, lambda, tmpy, tmpatax, tmpwx, tmpwtwx, r, p, tmpdot, tmpdotr, tmpdoti, d_xs, v_xs);
	complex2magtitude<<<dimGrid_x, dimBlock>>>(x_dc, xr_d, N);
//	complex2real<<<dimGrid_x, dimBlock>>>(x_dc, xr_d, xi_d, N);
	cudaMemcpy(xr, xr_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
//	cudaMemcpy(xi, xi_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
   
	cudaFree(tmpy);
	cudaFree(tmpatax);
	cudaFree(tmpwx);
	cudaFree(tmpwtwx);
	cudaFree(r);
	cudaFree(p);
	cudaFree(tmpdot);
	cudaFree(tmpdotr);
	cudaFree(tmpdoti);
	cudaFree(xr_d);
	cudaFree(xi_d);
	cudaFree(yr_d);
	cudaFree(yi_d);
	cudaFree(xpr_d);
	cudaFree(xpi_d);
	cudaFree(mask_d);
	cudaFree(x_dc);
	cudaFree(y_dc);
	cudaFree(xp_dc);
	cudaFree(b_dc);
	cudaFree(d_xs);
	cudaFree(v_xs);



}




