import ctypes
from ctypes import *
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
#import copy
#__all__ = [
#    "Wxs",
#    "adjDx",
#    "adjDy",
#    "Wtxs",
#    "Sxs",
#    "fft2c",
#    "ifft2c",
#    "Amx",
#    "Atmx",
#    "conj_grad",
#    "genkspacdata",
#    "ctrecon_admm"
#]


#xraytf = ctypes.CDLL('./libxraytransform.so', mode=ctypes.RTLD_GLOBAL)
#






class ctoplib():
    def __init__(self,nx=256,ny=256,nd=512,nv=668,SO=1000,OD=500,offset=0,mu=0.6,rho=0.0001,lam=0.08,N_iter=20,CG_iter=20,CG_tol=1e-8,Min_iter=10):    
#        self.nx=np.uint32(256)
#        self.ny=nx
#        
#        self.dx=float(250.0/nx);
#        self.dy=dx;
        self.nx=np.uint32(nx)
        self.ny=np.uint32(ny)        
        self.nv=np.uint32(nv)
        self.SO=float(SO)
        self.OD=float(OD)
        self.nd=np.uint32(nd)
#        self.nt=np.uint32(nt)
#        self.dy_det=float(0.388*1024/nd)
#        self.sd_phi=2*np.pi/nv*np.arange(0,nv,dtype= np.float32)
        self.offset=np.uint32(offset)
        
        
        self.mu=float(mu)
        self.rho=float(rho)
        if self.mu>0.000001:
            self.lam=self.mu/float(lam)
        else:
            self.lam=float(0.08)
        self.N_iter=np.uint32(N_iter)
        self.CG_maxit=np.uint32(CG_iter)
        self.CG_tol=float(CG_tol)
        self.Min_iter=np.uint32(Min_iter)
    def set_rho(self,rho):
        self.rho = rho        
    def Ax(self,im):      
#        im_pointer = im.ctypes.data_as(POINTER(c_float))
#        Y_pointer = Y.ctypes.data_as(POINTER(c_float))
#        sd_phi_pointer = self.sd_phi.ctypes.data_as(POINTER(c_float))
#        y_det_pointer = self.y_det.ctypes.data_as(POINTER(c_float))
#        id_X_pointer = self.id_X.ctypes.data_as(POINTER(c_uint32))
        
    
#        nx=np.uint32(256)
#        ny=nx
#        nt=np.uint32(im.size/(self.nx*self.ny))
        dx=float(250.0/self.nx);
        dy=dx
        nt=np.uint32(1)
#        nv=668;
#        SO=float(1000)
#        OD=float(500)
#        nd=np.uint32(512)
        dy_det=float(0.48*1024/self.nd)
        sd_phi=2*np.pi/self.nv*np.arange(0,self.nv,dtype= np.float32)
        y_os=float(self.offset*dy_det);
        y_det=(np.arange(-float(self.nd)/2.0,float(self.nd)/2.0,dtype= np.float32)+0.5)*dy_det+y_os
        y_det2=np.arange(-float(self.nd)/2.0,float(self.nd)/2.0+1.0,dtype= np.float32)*dy_det+y_os
        
        scale=dx
        SO=self.SO/scale
        OD=self.OD/scale
        y_det=y_det/scale
        dy_det=dy_det/scale
        y_os=y_os/scale
        
        
        nt=np.uint32(1);
        #X0=x0(:);
        
        Id_v=np.arange(0,self.nv,dtype= np.uint32)
        id_X=np.zeros_like(Id_v)
        Nv = self.nv*np.ones_like(Id_v)
        
        tmp_size=np.max(Nv)
        id_Y=Id_v
        Y=np.zeros((self.nv,self.nd), dtype=np.float32)
        
        im_pointer = im.ctypes.data_as(POINTER(c_float))
        Y_pointer = Y.ctypes.data_as(POINTER(c_float))
        sd_phi_pointer = sd_phi.ctypes.data_as(POINTER(c_float))
        y_det_pointer = y_det.ctypes.data_as(POINTER(c_float))
        id_X_pointer = id_X.ctypes.data_as(POINTER(c_uint32))
        ll = ctypes.cdll.LoadLibrary 
        xraytf = ll("./CT_Operators/libctrecon.so")  
        Axc = xraytf.Ax_fan_mf_gpu_new
        
        Axc.argtypes = [POINTER(c_float), POINTER(c_float), c_float, c_float, c_float, c_uint32, c_uint32, c_uint32, POINTER(c_float), c_uint32, POINTER(c_float), POINTER(c_uint32), c_uint32]
        Axc(im_pointer,Y_pointer, SO, OD, scale, self.nx, self.ny, self.nv,sd_phi_pointer, self.nd,y_det_pointer,id_X_pointer, nt)

        return Y.reshape(self.nv,self.nd)

    def Atx(self, Y):

        
        
        
        nt=np.uint32(1)
        dx=float(250.0/self.nx);
        dy=dx;
        nd=np.uint32(self.nd);
        dy_det=float(0.48*1024/self.nd)
        sd_phi=2*np.pi/self.nv*np.arange(0,self.nv,dtype= np.float32)
        y_os=float(0*dy_det);
        y_det=(np.arange(-float(self.nd)/2.0,float(self.nd)/2.0,dtype= np.float32)+0.5)*dy_det+y_os
        y_det2=np.arange(-float(self.nd)/2.0,float(self.nd)/2.0+1.0,dtype= np.float32)*dy_det+y_os
        
        scale=dx
        SO=self.SO/scale
        OD=self.OD/scale
        y_det=y_det/scale
        dy_det=dy_det/scale
        y_os=y_os/scale               
        
        Id_v=np.arange(0,self.nv,dtype= np.uint32)
        id_X=np.zeros_like(Id_v)
        Nv = self.nv*np.ones_like(Id_v)
        
        tmp_size=np.max(Nv)
        id_Y=Id_v
        Y_pointer = Y.ctypes.data_as(POINTER(c_float))
        sd_phi_pointer = sd_phi.ctypes.data_as(POINTER(c_float))
        y_det_pointer = y_det.ctypes.data_as(POINTER(c_float))
        id_X_pointer = id_X.ctypes.data_as(POINTER(c_uint32))    
        cos_phi=np.cos(sd_phi)
        sin_phi=np.sin(sd_phi)     
        angle_det=np.arctan2(y_det,SO+OD)
        cos_det=np.cos(angle_det)
        sin_det=np.sin(angle_det)
        
        Nv_pointer = Nv.ctypes.data_as(POINTER(c_uint32))
        id_Y_pointer = id_Y.ctypes.data_as(POINTER(c_uint32))
        cos_phi_pointer = cos_phi.ctypes.data_as(POINTER(c_float))
        sin_phi_pointer = sin_phi.ctypes.data_as(POINTER(c_float))
        angle_det_pointer = angle_det.ctypes.data_as(POINTER(c_float))
        cos_det_pointer = cos_det.ctypes.data_as(POINTER(c_float))
        sin_det_pointer = sin_det.ctypes.data_as(POINTER(c_float))
        
        aty=np.zeros((self.nx,self.ny), dtype=np.float32)
        aty_pointer = aty.ctypes.data_as(POINTER(c_float))
        
        ll = ctypes.cdll.LoadLibrary 
        xraytf = ll("./CT_Operators/libctrecon.so") 
        Atx = xraytf.Atx_fan_mf_gpu_new
        Atx.argtypes = [POINTER(c_float), POINTER(c_float), c_float, c_float, c_float, c_uint32, c_uint32, POINTER(c_float), c_uint32, POINTER(c_float),
                        c_float, c_float, c_uint32, POINTER(c_uint32), POINTER(c_uint32), c_uint32, c_uint32,
                        POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float)]        
        Atx(aty_pointer, Y_pointer, SO, OD, scale, self.nx, self.ny, sd_phi_pointer, self.nd, y_det_pointer, dy_det, y_os, nt, id_Y_pointer, Nv_pointer, tmp_size, 
            self.nv, cos_phi_pointer, sin_phi_pointer, cos_det_pointer, sin_det_pointer)
        
        return aty#.reshape(nt,self.nx,self.ny)



    def recon_cg(self, Y, Xp):
        
        
        nt=np.uint32(1)
        dx=float(250.0/self.nx);
        dy=dx;
        nd=np.uint32(self.nd);
        dy_det=float(0.48*1024/self.nd)
        sd_phi=2*np.pi/self.nv*np.arange(0,self.nv,dtype= np.float32)
        y_os=float(0*dy_det);
        y_det=(np.arange(-float(self.nd)/2.0,float(self.nd)/2.0,dtype= np.float32)+0.5)*dy_det+y_os
        y_det2=np.arange(-float(self.nd)/2.0,float(self.nd)/2.0+1.0,dtype= np.float32)*dy_det+y_os
        
        scale=dx
        SO=self.SO/scale
        OD=self.OD/scale
        y_det=y_det/scale
        dy_det=dy_det/scale
        y_os=y_os/scale               
        
        Id_v=np.arange(0,self.nv,dtype= np.uint32)
        id_X=np.zeros_like(Id_v)
        Nv = self.nv*np.ones_like(Id_v)
        
        tmp_size=np.max(Nv)
        id_Y=Id_v
#        Y_pointer = Y.ctypes.data_as(POINTER(c_float))
        sd_phi_pointer = sd_phi.ctypes.data_as(POINTER(c_float))
        y_det_pointer = y_det.ctypes.data_as(POINTER(c_float))
        id_X_pointer = id_X.ctypes.data_as(POINTER(c_uint32))    
        cos_phi=np.cos(sd_phi)
        sin_phi=np.sin(sd_phi)     
        angle_det=np.arctan2(y_det,SO+OD)
        cos_det=np.cos(angle_det)
        sin_det=np.sin(angle_det)
        
        Nv_pointer = Nv.ctypes.data_as(POINTER(c_uint32))
        id_Y_pointer = id_Y.ctypes.data_as(POINTER(c_uint32))
        cos_phi_pointer = cos_phi.ctypes.data_as(POINTER(c_float))
        sin_phi_pointer = sin_phi.ctypes.data_as(POINTER(c_float))
        angle_det_pointer = angle_det.ctypes.data_as(POINTER(c_float))
        cos_det_pointer = cos_det.ctypes.data_as(POINTER(c_float))
        sin_det_pointer = sin_det.ctypes.data_as(POINTER(c_float))
        
        
        

        xr=np.zeros((self.nx,self.ny), dtype=np.float32)  
        xr_pointer = xr.ctypes.data_as(POINTER(c_float))
        Ytmp=np.zeros((self.nv,self.nd), dtype=np.float32)
        Ytmp[:,:]=Y[:,:]
        Yr_pointer = Ytmp.ctypes.data_as(POINTER(c_float))

        xp_pointer = Xp.ctypes.data_as(POINTER(c_float))


        ll = ctypes.cdll.LoadLibrary 
        tf = ll("./CT_Operators/libctrecon.so")  
        recon = tf.ctrecon_conjugate_grad        
        recon.argtypes = [POINTER(c_float),  POINTER(c_float), POINTER(c_float),  c_float, c_float,  c_float, c_uint32, c_uint32, 
                          POINTER(c_float), c_int32, POINTER(c_float), c_float, c_float, c_uint32,
                          POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32), c_uint32, c_uint32, POINTER(c_float),POINTER(c_float),POINTER(c_float),
                          POINTER(c_float), c_float, c_float, c_uint32, c_float, c_float]

        recon(xr_pointer, Yr_pointer, xp_pointer, SO, OD, scale, self.nx, self.ny, 
              sd_phi_pointer, self.nd, y_det_pointer, dy_det, y_os, nt, 
              id_X_pointer, id_Y_pointer, Nv_pointer, tmp_size, self.nv, cos_phi_pointer, sin_phi_pointer, cos_det_pointer, 
              sin_det_pointer, self.mu, self.rho,  self.CG_maxit, self.CG_tol, self.lam);

        return  xr#reshape(self.N,self.N)#+1.0j*xi#.reshape(nt,self.nv,self.nd)         


    def recon_admm(self, Y, Xp):
        
        nt=np.uint32(1)
        dx=float(250.0/self.nx);
        dy=dx;
        nd=np.uint32(self.nd);
        dy_det=float(0.48*1024/self.nd)
        sd_phi=2*np.pi/self.nv*np.arange(0,self.nv,dtype= np.float32)
        y_os=float(0*dy_det);
        y_det=(np.arange(-float(self.nd)/2.0,float(self.nd)/2.0,dtype= np.float32)+0.5)*dy_det+y_os
        y_det2=np.arange(-float(self.nd)/2.0,float(self.nd)/2.0+1.0,dtype= np.float32)*dy_det+y_os
        
        scale=dx
        SO=self.SO/scale
        OD=self.OD/scale
        y_det=y_det/scale
        dy_det=dy_det/scale
        y_os=y_os/scale               
        
        Id_v=np.arange(0,self.nv,dtype= np.uint32)
        id_X=np.zeros_like(Id_v)
        Nv = self.nv*np.ones_like(Id_v)
        
        tmp_size=np.max(Nv)
        id_Y=Id_v
#        Y_pointer = Y.ctypes.data_as(POINTER(c_float))
        sd_phi_pointer = sd_phi.ctypes.data_as(POINTER(c_float))
        y_det_pointer = y_det.ctypes.data_as(POINTER(c_float))
        id_X_pointer = id_X.ctypes.data_as(POINTER(c_uint32))    
        cos_phi=np.cos(sd_phi)
        sin_phi=np.sin(sd_phi)     
        angle_det=np.arctan2(y_det,SO+OD)
        cos_det=np.cos(angle_det)
        sin_det=np.sin(angle_det)
        
        Nv_pointer = Nv.ctypes.data_as(POINTER(c_uint32))
        id_Y_pointer = id_Y.ctypes.data_as(POINTER(c_uint32))
        cos_phi_pointer = cos_phi.ctypes.data_as(POINTER(c_float))
        sin_phi_pointer = sin_phi.ctypes.data_as(POINTER(c_float))
        angle_det_pointer = angle_det.ctypes.data_as(POINTER(c_float))
        cos_det_pointer = cos_det.ctypes.data_as(POINTER(c_float))
        sin_det_pointer = sin_det.ctypes.data_as(POINTER(c_float))
        
        

        xr=np.zeros((self.nx,self.ny), dtype=np.float32)     
        Ytmp=np.zeros((self.nv,self.nd), dtype=np.float32)
        Ytmp[:,:]=Y[:,:]
        Y_pointer = Ytmp.ctypes.data_as(POINTER(c_float))
        xp_pointer = Xp.ctypes.data_as(POINTER(c_float))
#        xpi_pointer = Xp[1,:,:].ctypes.data_as(POINTER(c_float))
        xr_pointer = xr.ctypes.data_as(POINTER(c_float))
#        xi_pointer = xi.ctypes.data_as(POINTER(c_float))
        ll = ctypes.cdll.LoadLibrary 
        tf = ll("./CT_Operators/libctrecon.so")  
        recon = tf.ctrecon_admm       
        recon.argtypes = [POINTER(c_float),  POINTER(c_float), POINTER(c_float),  c_float, c_float,  c_float, c_uint32, c_uint32, 
                          POINTER(c_float), c_int32, POINTER(c_float), c_float, c_float, c_uint32,
                          POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32), c_uint32, c_uint32, POINTER(c_float),POINTER(c_float),POINTER(c_float),
                          POINTER(c_float), c_float, c_float, c_uint32,c_uint32, c_float, c_float]

        recon(xr_pointer, Y_pointer, xp_pointer, SO, OD, scale, self.nx, self.ny, 
              sd_phi_pointer, self.nd, y_det_pointer, dy_det, y_os, nt, 
              id_X_pointer, id_Y_pointer, Nv_pointer, tmp_size, self.nv, cos_phi_pointer, sin_phi_pointer, cos_det_pointer, 
              sin_det_pointer, self.mu, self.rho,  self.CG_maxit, self.N_iter, self.CG_tol, self.lam);

        return  xr#+1.0j*xi#.reshape(nt,self.nv,self.nd)         



    def gensindata(self, Xg, noiselevel=0.02):
        if len(Xg.shape) == 2:
            Xg = np.stack([Xg],axis=0)
        Y=np.zeros((Xg.shape[0],self.nv,self.nd),dtype=np.float32)
        Xgn=np.zeros((self.nx,self.ny),dtype=np.float32)       
        for i in range(Xg.shape[0]):
            Xgn[:,:]=Xg[i,:,:]+noiselevel*(np.random.standard_normal((self.nx, self.ny)))
            Y[i,:,:]=self.Ax(Xgn)
        return Y
    def recon_cg_batch(self, Y, Xp):
        if len(Y.shape) == 2:
            Y = np.stack([Y],axis=0)
            Xp = np.stack([Xp],axis=0)
        X=np.zeros((Y.shape[0],self.nx,self.ny),dtype=np.float32) 
        tmp=np.zeros((self.nx,self.ny),dtype=np.float32)        
        for i in range(Y.shape[0]):
            tmp[:,:]=Xp[i,:,:]            
            X[i,:,:]=self.recon_cg(Y[i,:,:], tmp)
        return X
    def recon_admm_batch(self, Y, Xp):
        if len(Y.shape) ==  2:
            Y = np.stack([Y],axis=0)
            Xp = np.stack([Xp],axis=0)
        X=np.zeros((Y.shape[0],self.nx,self.ny),dtype=np.float32)  
        tmp=np.zeros((self.nx,self.ny),dtype=np.float32)
        for i in range(Y.shape[0]):
            tmp[:,:]=Xp[i,:,:]
            X[i,:,:]=self.recon_admm(Y[i,:,:],tmp)#tmp Xp[i,:,:]
        return X








def Wxs(im):
    Dx = np.zeros_like(im)
    Dx[:-1,:] = im[1:,:]-im[:-1,:]
    Dy = np.zeros_like(im)
    Dy[:, :-1] = im[:,1:]-im[:,:-1]
    res=np.concatenate((Dx.reshape((Dx.shape[0],Dx.shape[1],1)),Dy.reshape((Dy.shape[0],Dy.shape[1],1))),axis=2)
    return res
    

def adjDx(x):
    Dtx=np.zeros_like(x)
    Dtx[1:,:]=x[:-1,:] - x[1:,:]
    Dtx[0,:]=-x[0,:]
    Dtx[-1,:]=x[-2,:]
    return Dtx

def adjDy(x):
    Dty=np.zeros_like(x)
    Dty[:,1:]=x[:,:-1] - x[:,1:]
    Dty[:,0]=-x[:,0]
    Dty[:,-1]=x[:,-2]
    return Dty

def Wtxs(y):
    res=adjDx(y[:,:,0]) + adjDy(y[:,:,1])
    return res

def Sxs(Bub,lam,mu):
    res = np.zeros_like(Bub)
    s=np.sqrt(Bub[:,:,0]*Bub[:,:,0].conj() + Bub[:,:,1]*Bub[:,:,1].conj())
    res[:,:,0]=np.maximum(s-mu/lam,0)*Bub[:,:,0]/(s+np.spacing(1))
    res[:,:,1]=np.maximum(s-mu/lam,0)*Bub[:,:,1]/(s+np.spacing(1))
    return res

def fft2c(x):
    res=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    return res

def ifft2c(x):
    res=np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x)))
    return res    



class mritf_cpu():
    def __init__(self,mask=None): 
        self.mask=mask
    def Ax(self,x):
        res=self.mask*fft2c(x)
        return res    
    def Atx(self,y):
        res=ifft2c(y*self.mask)
        return res




def Amx(x,mask):
    res=mask*fft2c(x)
    return res

def Atmx(y,mask):
    res=ifft2c(y*mask)
    return res

def Atmxs(Y,mask):
    AtY=np.zeros_like(Y)
    if len(Y.shape) == 2:
        Y = np.stack([Y],axis=0)
    for i in range(Y.shape[0]):
        AtY[i,:,:]=Atmx(Y[i,:,:],mask)
    return AtY

#def conj_grad(mask,b, mu,rho,x,maxit,CG_tol0):
#    r = b - Atmx(Amx(x,mask),mask)-mu*Wtxs(Wxs(x))-rho*x
#    p = r
#    rsold = np.inner(r.flatten(),r.flatten().conj())
#    CG_tol=rsold*CG_tol0
#    for i in range(maxit):
#        Ap=Atmx(Amx(p,mask),mask)+mu*Wtxs(Wxs(p))+rho*p
#        alpha=rsold/np.inner(p.flatten(),Ap.flatten().conj())
#        x=x+alpha*p
#        r=r-alpha*Ap
#        rsnew=np.inner(r.flatten(),r.flatten().conj())
#        if np.sqrt(rsnew)<CG_tol:
#            break
#        p=r+(rsnew/rsold)*p
#        rsold=rsnew        
#    cg_err=np.sqrt(np.abs(rsnew))
#    cg_n=i
#    return x,cg_err,cg_n
    
def genkspacdata(Xg, mask, noiselevel=0.02):
    if len(Xg.shape) == 2:
        Xg = np.stack([Xg],axis=0)
    Y=np.zeros_like(Xg)+0.0j*np.zeros_like(Xg)
    for i in range(Xg.shape[0]):
        Y[i,:,:]=Amx(Xg[i,:,:]+noiselevel*(np.random.standard_normal((Xg.shape[1],Xg.shape[2]))+1.0j*np.random.standard_normal((Xg.shape[1],Xg.shape[2]))),mask)
    return Y


class param():
    def __init__(self,mu=0.6,rho=0.0001,lam=0.08,N_iter=20,cg_iter=20,CG_tol=1e-8,Min_iter=10):    
        self.mu=mu
        self.rho=rho
        if self.mu>0.000001:
            self.lam=self.mu/0.08
        else:
            self.lam=0.08
        self.N_iter=N_iter
        self.cg_iter=cg_iter
        self.CG_tol=CG_tol
        self.Min_iter=Min_iter
#        self.mask=mask

#def mrirecon_admm(X, XplusB, Y, M, para):
#    assert(X.shape == XplusB.shape == Y.shape)
#    if len(X.shape) == 2:
#        X = np.stack([X],axis=0)
#        XplusB = np.stack([XplusB],axis=0)
#        Y = np.stack([Y],axis=0)
#        
#    Xr=np.zeros_like(X)
#    for i in range(X.shape[0]):        
#        Xi=X[i,:,:]
#    #    Xt=np.zeros_like(x0)
#        d_xs=np.zeros_like(Wxs(Xi));
#        v_xs=np.zeros_like(Wxs(Xi));
#        for n_iter in range(para.N_iter):
#            g=M.Atx(Y[i,:,:])+para.mu*Wtxs(d_xs+v_xs)+para.rho*XplusB[i,:,:]
#            [Xi,cg_err,cg_n]=conj_grad_mri(M, g, para.mu,para.rho,Xi,para.cg_iter,para.CG_tol)
#            print('%d -- CG: N= %d   error= %.5f' %(n_iter,cg_n,cg_err))
#            temp_xs=Wxs(Xi)-v_xs
#            d_xs=Sxs(temp_xs, para.lam, para.mu)
#            v_xs=d_xs-temp_xs
#            
#            if cg_err>100000:
#                break
#            
#        Xr[i,:,:]=np.abs(Xi)
#    return Xr
#
#def conj_grad_mri(M, b, mu, rho, x, maxit, CG_tol0):
#    r = b - M.Atx(M.Ax(x))-mu*Wtxs(Wxs(x))-rho*x
#    p = r
#    rsold = np.inner(r.flatten(),r.flatten().conj())
#    CG_tol=rsold*CG_tol0
#    for i in range(maxit):
#        Ap=M.Atx(M.Ax(p))+mu*Wtxs(Wxs(p))+rho*p
#        alpha=rsold/np.inner(p.flatten(),Ap.flatten().conj())
#        x=x+alpha*p
#        r=r-alpha*Ap
#        rsnew=np.inner(r.flatten(), r.flatten().conj())
#        if np.sqrt(rsnew)<CG_tol:
#            break
#        p=r+(rsnew/rsold)*p
#        rsold=rsnew        
#    cg_err=np.sqrt(np.abs(rsnew))
#    cg_n=i
#    return x,cg_err,cg_n
    
def conj_grad(M,b, mu,rho,x,maxit,CG_tol0):
    r = b - M.Atx(M.Ax(x))-mu*Wtxs(Wxs(x))-rho*x
    p = r
    rsold = np.inner(r.flatten(),r.flatten().conj())
    CG_tol=rsold*CG_tol0
    for i in range(maxit):
        Ap=M.Atx(M.Ax(p))+mu*Wtxs(Wxs(p))+rho*p
        alpha=rsold/np.inner(p.flatten(),Ap.flatten().conj())
        x=x+alpha*p
        r=r-alpha*Ap
        rsnew=np.inner(r.flatten(),r.flatten().conj())
        if np.sqrt(rsnew)<CG_tol:
            break
        p=r+(rsnew/rsold)*p
        rsold=rsnew        
    cg_err=np.sqrt(np.abs(rsnew))
    cg_n=i
    return x,cg_err,cg_n
def mrirecon_admm(X, XplusB, Y, M, para):
    assert(X.shape == XplusB.shape )#== Y.shape)
    if len(X.shape) == 2:
        X = np.stack([X],axis=0)
        XplusB = np.stack([XplusB],axis=0)
        Y = np.stack([Y],axis=0)
        
    Xr=np.zeros_like(X)
    for i in range(X.shape[0]):        
        Xi=X[i,:,:]
    #    Xt=np.zeros_like(x0)
        d_xs=np.zeros_like(Wxs(Xi));
        v_xs=np.zeros_like(Wxs(Xi));
        for n_iter in range(para.N_iter):
            g=M.Atx(Y[i,:,:])+para.mu*Wtxs(d_xs+v_xs)+para.rho*XplusB[i,:,:]
            [Xi,cg_err,cg_n]=conj_grad(M,g, para.mu,para.rho,Xi,para.cg_iter,para.CG_tol)
            print('%d -- CG: N= %d   error= %.5f' %(n_iter,cg_n,cg_err))
            temp_xs=Wxs(Xi)-v_xs
            d_xs=Sxs(temp_xs,para.lam,para.mu)
            v_xs=d_xs-temp_xs
            
            if cg_err>100000:
                break
            
        Xr[i,:,:]=abs(Xi)
    return Xr


def conj_grad_v2(mask,b,rho,x,maxit,CG_tol0):
    r = b - Atmx(Amx(x,mask),mask)-rho*x
    p = r
    rsold = np.inner(r.flatten(),r.flatten().conj())
    CG_tol=rsold*CG_tol0
    for i in range(maxit):
        Ap=Atmx(Amx(p,mask),mask)+rho*p
        alpha=rsold/np.inner(p.flatten(),Ap.flatten().conj())
        x=x+alpha*p
        r=r-alpha*Ap
        rsnew=np.inner(r.flatten(),r.flatten().conj())
        if np.sqrt(rsnew)<CG_tol:
            break
        p=r+(rsnew/rsold)*p
        rsold=rsnew        
    cg_err=np.sqrt(np.abs(rsnew))
    cg_n=i
    return x,cg_err,cg_n
def mrirecon_cg(X, XplusB, AtY, para):
    assert(X.shape == XplusB.shape == AtY.shape)
    if len(X.shape) == 2:
        X = np.stack([X],axis=0)
        XplusB = np.stack([XplusB],axis=0)
        AtY = np.stack([AtY],axis=0)
        
    Xr=np.zeros_like(X)
    for i in range(X.shape[0]):        
        Xi=X[i,:,:]
    #    Xt=np.zeros_like(x0)
#        d_xs=np.zeros_like(Wxs(Xi));
#        v_xs=np.zeros_like(Wxs(Xi));
#        for n_iter in range(para.N_iter):
        g=AtY[i,:,:]+para.rho*XplusB[i,:,:] #Atmx(Y[i,:,:],para.mask)
        [Xi,cg_err,cg_n]=conj_grad_v2(para.mask,g, para.rho,Xi,para.cg_iter,para.CG_tol)
#        print('CG: N= %d   error= %.5f' %(cg_n,cg_err))
#        temp_xs=Wxs(Xi)-v_xs
#        d_xs=Sxs(temp_xs,para.lam,para.mu)
#        v_xs=d_xs-temp_xs
            
#            if cg_err>10:
#                break
            
        Xr[i,:,:]=abs(Xi)
    return Xr



def recon_admm(X, XplusB, Y, M, para):
    assert(X.shape == XplusB.shape)
    if len(X.shape) == 2:
        X = np.stack([X],axis=0)
        XplusB = np.stack([XplusB],axis=0)
        Y = np.stack([Y],axis=0)
#    print(X.shape,Y.shape)    
    Xr=np.zeros_like(X)
    for i in range(X.shape[0]):        
        Xi=X[i,:,:]
    #    Xt=np.zeros_like(x0)
        d_xs=np.zeros_like(Wxs(Xi));
        v_xs=np.zeros_like(Wxs(Xi));
        for n_iter in range(para.N_iter):
#            plt.imshow(Y.reshape(668,512),plt.cm.gray)
    
            g=M.Atx(Y[i,:,:])+para.mu*Wtxs(d_xs+v_xs)+para.rho*XplusB[i,:,:]
#            plt.imshow(Y[i,:,:].reshape(668,512),plt.cm.gray)            
            [Xi,cg_err,cg_n]=conj_grad(M,para.mask,g, para.mu,para.rho,Xi,para.cg_iter,para.CG_tol)
#            plt.imshow(Xi.reshape(256,256),plt.cm.gray)
#            print(g.dtype, Xi.dtype)
            print('%d -- CG: N= %d   error= %.5f' %(n_iter,cg_n,cg_err))
            temp_xs=Wxs(Xi)-v_xs
            d_xs=Sxs(temp_xs,para.lam,para.mu)
            v_xs=d_xs-temp_xs
            
            if cg_err>100000:
                break
            
        Xr[i,:,:]=abs(Xi)
    return Xr



def mrirecon_zf(X, Y, para, pdf):
    Xr=np.zeros_like(X)
    for i in range(X.shape[0]):  
        Xr[i,:,:]=abs(Atmx(Y[i,:,:]/pdf, para.mask)) #/pdf
    
    
    return Xr
    


        
        
    
















