#import sys
#sys.path.append('../MRI_Operators/')
import ctypes
from ctypes import *
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import copy
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






class mriroplib():
    def __init__(self,mask=None, N=192, mu=0.6,rho=0.0001,lam=0.08,N_iter=20,CG_iter=20,CG_tol=1e-8,Min_iter=10):    

        self.mask=mask.astype(np.float32)
        self.N=np.uint32(N)
        
        self.mu=float(mu)
        self.rho=float(rho)
        if self.mu>0.000001:
            self.lam=self.mu/float(0.08)
        else:
            self.lam=float(0.08)
        self.N_iter=np.uint32(N_iter)
        self.CG_maxit=np.uint32(CG_iter)
        self.CG_tol=float(CG_tol)
        self.Min_iter=np.uint32(Min_iter)
        
    def Ax(self,im):
#        Yr=np.zeros((self.N,self.N), dtype=np.float32)
#        Yi=np.zeros((self.N,self.N), dtype=np.float32)
#        Yr=Y[1.:,:]
#        Yi=Y[1.:,:]
        Y=np.zeros((2,self.N,self.N), dtype=np.float32)
        Yr_pointer = Y[0,:,:].ctypes.data_as(POINTER(c_float))
        Yi_pointer = Y[1,:,:].ctypes.data_as(POINTER(c_float))
        imr= copy.deepcopy(im.real)
        imi= copy.deepcopy(im.imag)
        imr_pointer = imr.ctypes.data_as(POINTER(c_float))
        imi_pointer = imi.ctypes.data_as(POINTER(c_float))
        mask_pointer = self.mask.ctypes.data_as(POINTER(c_float))
        ll = ctypes.cdll.LoadLibrary 
        tf = ll("./MRI_Operators/libmrirecon.so")  
        Axc = tf.Ax_mri        
        Axc.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint32]
        Axc(Yr_pointer, Yi_pointer, imr_pointer, imi_pointer,mask_pointer, self.N)
#        Y[0,:,:,]=Y[0,:,:,].T
#        Y[1,:,:,]=Y[1,:,:,].T
        return  Y#(Yr,Yi)#Yr+1.0j*Yi#.reshape(nt,self.nv,self.nd)
    def AxC(self,im):
#        Yr=np.zeros((self.N,self.N), dtype=np.float32)
#        Yi=np.zeros((self.N,self.N), dtype=np.float32)
#        Yr=Y[1.:,:]
#        Yi=Y[1.:,:]
        Y=np.zeros((2,self.N,self.N), dtype=np.float32)
        Yr_pointer = Y[0,:,:].ctypes.data_as(POINTER(c_float))
        Yi_pointer = Y[1,:,:].ctypes.data_as(POINTER(c_float))
#        imr= copy.deepcopy(im.real)
#        imi= copy.deepcopy(im.imag)
        
        imr_pointer = im[0,:,:].ctypes.data_as(POINTER(c_float))
        imi_pointer = im[1,:,:].ctypes.data_as(POINTER(c_float))
        mask_pointer = self.mask.ctypes.data_as(POINTER(c_float))
        ll = ctypes.cdll.LoadLibrary 
        tf = ll("./MRI_Operators/libmrirecon.so")  
        Axc = tf.Ax_mri        
        Axc.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint32]
        Axc(Yr_pointer, Yi_pointer, imr_pointer, imi_pointer,mask_pointer, self.N)
#        Y[0,:,:,]=Y[0,:,:,].T
#        Y[1,:,:,]=Y[1,:,:,].T
        return  Y#(Yr,Yi)#Yr+1.0j*Yi#.reshape(nt,self.nv,self.nd)

    def Atx(self, Y):

        atyr=np.zeros((self.N,self.N), dtype=np.float32)
        atyi=np.zeros((self.N,self.N), dtype=np.float32)
#        aty=np.zeros((2,self.N,self.N), dtype=np.float32)
#        Yr=np.transpose(Y,(1,0)).real
#        Yi=np.transpose(Y,(1,0)).imag
#        Yr=Y.real
#        Yi=Y.imag        
        Yr_pointer = Y[0,:,:].ctypes.data_as(POINTER(c_float))
        Yi_pointer = Y[1,:,:].ctypes.data_as(POINTER(c_float))
        mask_pointer = self.mask.ctypes.data_as(POINTER(c_float))
        atyr_pointer = atyr.ctypes.data_as(POINTER(c_float))
        atyi_pointer = atyi.ctypes.data_as(POINTER(c_float))
        ll = ctypes.cdll.LoadLibrary 
        tf = ll("./MRI_Operators/libmrirecon.so")  
        Axc = tf.Atx_mri        
        Axc.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint32]
        Axc(atyr_pointer, atyi_pointer, Yr_pointer, Yi_pointer, mask_pointer,self.N)
        return  atyr#+1.0j*atyi#.reshape(nt,self.nv,self.nd)        


    def recon_cg(self, Y, Xp):

        xr=np.zeros((self.N,self.N), dtype=np.float32)
#        xi=np.zeros((self.N,self.N), dtype=np.float32)
#        xr=np.zeros((self.N,self.N), dtype=np.float32)
#        xi=np.zeros((self.N,self.N), dtype=np.float32)
#        aty=np.zeros((2,self.N,self.N), dtype=np.float32)
#        Yr=np.transpose(Y,(1,0)).real
#        Yi=np.transpose(Y,(1,0)).imag
#        Yr=Y.real
#        Yi=Y.imag        
        Yr_pointer = Y[0,:,:].ctypes.data_as(POINTER(c_float))
        Yi_pointer = Y[1,:,:].ctypes.data_as(POINTER(c_float))
        xpr_pointer = Xp.ctypes.data_as(POINTER(c_float))
#        xpi_pointer = Xp[1,:,:].ctypes.data_as(POINTER(c_float))
        mask_pointer = self.mask.ctypes.data_as(POINTER(c_float))
        xr_pointer = xr.ctypes.data_as(POINTER(c_float))
#        xi_pointer = xi.ctypes.data_as(POINTER(c_float))
        ll = ctypes.cdll.LoadLibrary 
        tf = ll("./MRI_Operators/libmrirecon.so")  
        recon = tf.mrirecon_conjugate_grad        
        recon.argtypes = [POINTER(c_float),  POINTER(c_float), POINTER(c_float), POINTER(c_float), 
                          c_float, c_float, c_uint32, c_float, POINTER(c_float), c_uint32, c_float]

        recon(xr_pointer, Yr_pointer, Yi_pointer, xpr_pointer, self.mu, self.rho,  self.CG_maxit,  self.CG_tol, mask_pointer, self.N,  self.lam);

        return  xr#reshape(self.N,self.N)#+1.0j*xi#.reshape(nt,self.nv,self.nd)         


    def recon_admm(self, Y, Xp):

        xr=np.zeros((self.N,self.N), dtype=np.float32)
#        xi=np.zeros((self.N,self.N), dtype=np.float32)
#        xr=np.zeros((self.N,self.N), dtype=np.float32)
#        xi=np.zeros((self.N,self.N), dtype=np.float32)
#        aty=np.zeros((2,self.N,self.N), dtype=np.float32)
#        Yr=np.transpose(Y,(1,0)).real
#        Yi=np.transpose(Y,(1,0)).imag
#        Yr=Y.real
#        Yi=Y.imag        
        Yr_pointer = Y[0,:,:].ctypes.data_as(POINTER(c_float))
        Yi_pointer = Y[1,:,:].ctypes.data_as(POINTER(c_float))
        xpr_pointer = Xp.ctypes.data_as(POINTER(c_float))
#        xpi_pointer = Xp[1,:,:].ctypes.data_as(POINTER(c_float))
        mask_pointer = self.mask.ctypes.data_as(POINTER(c_float))
        xr_pointer = xr.ctypes.data_as(POINTER(c_float))
#        xi_pointer = xi.ctypes.data_as(POINTER(c_float))
        ll = ctypes.cdll.LoadLibrary 
        tf = ll("./MRI_Operators/libmrirecon.so")  
        recon = tf.mrirecon_admm       
        recon.argtypes = [POINTER(c_float),  POINTER(c_float), POINTER(c_float), POINTER(c_float),  
                          c_float, c_float, c_uint32, c_float, c_uint32, POINTER(c_float), c_uint32, c_float]

        recon(xr_pointer , Yr_pointer, Yi_pointer, xpr_pointer, self.mu, self.rho,  self.CG_maxit,  self.CG_tol, self.N_iter, mask_pointer, self.N,  self.lam);

        return  xr#+1.0j*xi#.reshape(nt,self.nv,self.nd)         

    def genksdata(self, Xg, noiselevel=0.02):
        if len(Xg.shape) == 2:
            Xg = np.stack([Xg],axis=0)
        Y=np.zeros((Xg.shape[0],2,self.N,self.N),dtype=np.float32)
        Xgn=np.zeros((2, self.N,self.N),dtype=np.float32)       
        for i in range(Xg.shape[0]):
            Xgn[0,:,:]=Xg[i,:,:]+noiselevel*(np.random.standard_normal((self.N, self.N)))
            Xgn[1,:,:]=noiselevel*(np.random.standard_normal((self.N, self.N)))
            Y[i,:,:,:]=self.AxC(Xgn)
        return Y
    def recon_cg_batch(self, Y, Xp):
        if len(Y.shape) == 3:
            Y = np.stack([Y],axis=0)
        X=np.zeros((Y.shape[0],self.N,self.N),dtype=np.float32) 
        tmp=np.zeros((self.N,self.N),dtype=np.float32)        
        for i in range(Y.shape[0]):
            tmp[:,:]=Xp[i,:,:]            
            X[i,:,:]=self.recon_cg(Y[i,:,:,:], tmp)
        return X
#    def recon_cg_batch(self, Y, Xp):
#        if len(Y.shape) == 3:
#            Y = np.stack([Y],axis=0)
#        X=np.zeros((Y.shape[0],self.N,self.N),dtype=np.float32)    
#        for i in range(Y.shape[0]):       
#            X[i,:,:]=self.recon_cg(Y[i,:,:,:], Xp[i,:,:]  )
#        return X
    def recon_admm_batch(self, Y, Xp):
        if len(Y.shape) == 3:
            Y = np.stack([Y],axis=0)
            # if
        X=np.zeros((Y.shape[0],self.N,self.N),dtype=np.float32)  
        tmp=np.zeros((self.N,self.N),dtype=np.float32)
        for i in range(Y.shape[0]):
            tmp[:,:]=Xp[i,:,:]
            X[i,:,:]=self.recon_admm(Y[i,:,:,:],tmp)#tmp
        return X
    
    def Ax_batch(self, X):
        if len(X.shape) == 2:
            X = np.stack([X],axis=0)
        Y=np.zeros((X.shape[0],2,self.N,self.N),dtype=np.float32)  
        tmp=np.zeros((self.N,self.N),dtype=np.float32)
        for i in range(Y.shape[0]):   
            tmp = X[i,:,:]
            Y[i,:,:,:]=self.Ax(tmp)
        return Y
    def Atx_batch(self, Y):
        if len(Y.shape) == 3:
            Y = np.stack([Y],axis=0) 
        X=np.zeros((Y.shape[0],self.N,self.N),dtype=np.float32)
        for i in range(Y.shape[0]):         
            X[i,:,:]=self.Atx(Y[i,:,:,:])
        return X

        
#
#    def recon_admm_batch(self, Y, Xp):
#        if len(Y.shape) == 3:
#            Y = np.stack([Y],axis=0)
#        X=np.zeros((Y.shape[0],self.N,self.N),dtype=np.float32)  
#        for i in range(Y.shape[0]):
#            X[i,:,:]=self.recon_admm(Y[i,:,:,:],Xp[i,:,:])#tmp
#        return X









class mriroplibC():
    def __init__(self,mask=None, N=192, mu=0.6,rho=0.0001,lam=0.08,N_iter=20,CG_iter=20,CG_tol=1e-8,Min_iter=10):    

        self.mask=mask.astype(np.float32)
        self.N=np.uint32(N)
        
        self.mu=float(mu)
        self.rho=float(rho)
        if self.mu>0.000001:
            self.lam=self.mu/float(0.08)
        else:
            self.lam=float(0.08)
        self.N_iter=np.uint32(N_iter)
        self.CG_maxit=np.uint32(CG_iter)
        self.CG_tol=float(CG_tol)
        self.Min_iter=np.uint32(Min_iter)
        
    def Ax(self,im):
#        Yr=np.zeros((self.N,self.N), dtype=np.float32)
#        Yi=np.zeros((self.N,self.N), dtype=np.float32)
#        Yr=Y[1.:,:]
#        Yi=Y[1.:,:]
        Y=np.zeros((2,self.N,self.N), dtype=np.float32)
        Yr_pointer = Y[0,:,:].ctypes.data_as(POINTER(c_float))
        Yi_pointer = Y[1,:,:].ctypes.data_as(POINTER(c_float))
        imr= copy.deepcopy(im.real)
        imi= copy.deepcopy(im.imag)
        imr_pointer = imr.ctypes.data_as(POINTER(c_float))
        imi_pointer = imi.ctypes.data_as(POINTER(c_float))
        mask_pointer = self.mask.ctypes.data_as(POINTER(c_float))
        ll = ctypes.cdll.LoadLibrary 
        tf = ll("./MRI_Operators/libmrireconC.so")  
        Axc = tf.Ax_mri        
        Axc.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint32]
        Axc(Yr_pointer, Yi_pointer, imr_pointer, imi_pointer,mask_pointer, self.N)
#        Y[0,:,:,]=Y[0,:,:,].T
#        Y[1,:,:,]=Y[1,:,:,].T
        return  Y#(Yr,Yi)#Yr+1.0j*Yi#.reshape(nt,self.nv,self.nd)

    def Atx(self, Y):

        atyr=np.zeros((self.N,self.N), dtype=np.float32)
        atyi=np.zeros((self.N,self.N), dtype=np.float32)
#        aty=np.zeros((2,self.N,self.N), dtype=np.float32)
#        Yr=np.transpose(Y,(1,0)).real
#        Yi=np.transpose(Y,(1,0)).imag
#        Yr=Y.real
#        Yi=Y.imag        
        Yr_pointer = Y[0,:,:].ctypes.data_as(POINTER(c_float))
        Yi_pointer = Y[1,:,:].ctypes.data_as(POINTER(c_float))
        mask_pointer = self.mask.ctypes.data_as(POINTER(c_float))
        atyr_pointer = atyr.ctypes.data_as(POINTER(c_float))
        atyi_pointer = atyi.ctypes.data_as(POINTER(c_float))
        ll = ctypes.cdll.LoadLibrary 
        tf = ll("./MRI_Operators/libmrireconC.so")  
        Axc = tf.Atx_mri        
        Axc.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), c_uint32]
        Axc(atyr_pointer, atyi_pointer, Yr_pointer, Yi_pointer, mask_pointer,self.N)
        return  atyr#+1.0j*atyi#.reshape(nt,self.nv,self.nd)        


    def recon_cg(self, Y, Xp):

        xr=np.zeros((self.N,self.N), dtype=np.float32)
        xi=np.zeros((self.N,self.N), dtype=np.float32)
#        xr=np.zeros((self.N,self.N), dtype=np.float32)
#        xi=np.zeros((self.N,self.N), dtype=np.float32)
#        aty=np.zeros((2,self.N,self.N), dtype=np.float32)
#        Yr=np.transpose(Y,(1,0)).real
#        Yi=np.transpose(Y,(1,0)).imag
#        Yr=Y.real
#        Yi=Y.imag        
        Yr_pointer = Y[0,:,:].ctypes.data_as(POINTER(c_float))
        Yi_pointer = Y[1,:,:].ctypes.data_as(POINTER(c_float))
        xpr_pointer = Xp[0,:,:].ctypes.data_as(POINTER(c_float))
        xpi_pointer = Xp[1,:,:].ctypes.data_as(POINTER(c_float))
        mask_pointer = self.mask.ctypes.data_as(POINTER(c_float))
        xr_pointer = xr.ctypes.data_as(POINTER(c_float))
        xi_pointer = xi.ctypes.data_as(POINTER(c_float))
        ll = ctypes.cdll.LoadLibrary 
        tf = ll("./MRI_Operators/libmrireconC.so")  
        recon = tf.mrirecon_conjugate_grad        
        recon.argtypes = [POINTER(c_float),  POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
                          c_float, c_float, c_uint32, c_float, POINTER(c_float), c_uint32, c_float]

        recon(xr_pointer, xi_pointer,Yr_pointer, Yi_pointer, xpr_pointer,xpi_pointer, self.mu, self.rho,  self.CG_maxit,  self.CG_tol, mask_pointer, self.N,  self.lam);

        return  xr.T#reshape(self.N,self.N)#+1.0j*xi#.reshape(nt,self.nv,self.nd)         


    def recon_admm(self, Y, Xp):

        xr=np.zeros((self.N,self.N), dtype=np.float32)
        xi=np.zeros((self.N,self.N), dtype=np.float32)

#        xi=np.zeros((self.N,self.N), dtype=np.float32)
#        aty=np.zeros((2,self.N,self.N), dtype=np.float32)
#        Yr=np.transpose(Y,(1,0)).real
#        Yi=np.transpose(Y,(1,0)).imag
#        Yr=Y.real
#        Yi=Y.imag        
        Yr_pointer = Y[0,:,:].ctypes.data_as(POINTER(c_float))
        Yi_pointer = Y[1,:,:].ctypes.data_as(POINTER(c_float))
        xpr_pointer = Xp[0,:,:].ctypes.data_as(POINTER(c_float))
        xpi_pointer = Xp[1,:,:].ctypes.data_as(POINTER(c_float))
        mask_pointer = self.mask.ctypes.data_as(POINTER(c_float))
        xr_pointer = xr.ctypes.data_as(POINTER(c_float))
        xi_pointer = xi.ctypes.data_as(POINTER(c_float))
        ll = ctypes.cdll.LoadLibrary 
        tf = ll("./MRI_Operators/libmrireconC.so")  
        recon = tf.mrirecon_admm       
        recon.argtypes = [POINTER(c_float),  POINTER(c_float), POINTER(c_float), POINTER(c_float),  POINTER(c_float), POINTER(c_float),
                          c_float, c_float, c_uint32, c_float, c_uint32, POINTER(c_float), c_uint32, c_float]

        recon(xr_pointer , xi_pointer, Yr_pointer, Yi_pointer, xpr_pointer, xpi_pointer, self.mu, self.rho,  self.CG_maxit,  self.CG_tol, self.N_iter, mask_pointer, self.N,  self.lam);

        return  xr.T#+1.0j*xi#.reshape(nt,self.nv,self.nd)         









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
    


        
        
    
















