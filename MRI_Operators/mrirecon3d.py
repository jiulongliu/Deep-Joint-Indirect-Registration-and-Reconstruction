# -*- coding: utf-8 -*-
"""
Created on Sun Apr 09 09:52:41 2017

@author: Jiulong Liu
"""


import numpy as np

__all__ = [
    "Wxs",
    "adjDx",
    "adjDy",
    "Wtxs",
    "Sxs",
    "fft2c",
    "ifft2c",
    "Amx",
    "Atmx",
    "conj_grad",
    "genkspacdata",
    "mrirecon_admm"
]
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

def conj_grad(mask,b, mu,rho,x,maxit,CG_tol0):
    r = b - Atmx(Amx(x,mask),mask)-mu*Wtxs(Wxs(x))-rho*x
    p = r
    rsold = np.inner(r.flatten(),r.flatten().conj())
    CG_tol=rsold*CG_tol0
    for i in range(maxit):
        Ap=Atmx(Amx(p,mask),mask)+mu*Wtxs(Wxs(p))+rho*p
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
    
def genkspacdata(Xg,mask,noiselevel=0.02):
    if len(Xg.shape) == 2:
        Xg = np.stack([Xg],axis=0)
    Y=np.zeros_like(Xg)+0.0j*np.zeros_like(Xg)
    for i in range(Xg.shape[0]):
        Y[i,:,:]=Amx(Xg[i,:,:]+noiselevel*(np.random.standard_normal((Xg.shape[1],Xg.shape[2]))+1.0j*np.random.standard_normal((Xg.shape[1],Xg.shape[2]))),mask)
    return Y


class param():
    def __init__(self,mu=0.1,rho=0.5,lam=0.08,N_iter=20,cg_iter=20,CG_tol=1e-5,Min_iter=5,mask=None):    
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
        self.mask=mask

def mrirecon_admm(X, XplusB, Y, para):
    assert(X.shape == XplusB.shape == Y.shape)
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
            g=Atmx(Y[i,:,:],para.mask)+para.mu*Wtxs(d_xs+v_xs)+para.rho*XplusB[i,:,:]
            [Xi,cg_err,cg_n]=conj_grad(para.mask,g, para.mu,para.rho,Xi,para.cg_iter,para.CG_tol)
#            print('%d -- CG: N= %d   error= %.5f' %(n_iter,cg_n,cg_err))
            temp_xs=Wxs(Xi)-v_xs
            d_xs=Sxs(temp_xs,para.lam,para.mu)
            v_xs=d_xs-temp_xs
            
            if cg_err>10:
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
        print('CG: N= %d   error= %.5f' %(cg_n,cg_err))
#        temp_xs=Wxs(Xi)-v_xs
#        d_xs=Sxs(temp_xs,para.lam,para.mu)
#        v_xs=d_xs-temp_xs
            
#            if cg_err>10:
#                break
            
        Xr[i,:,:]=abs(Xi)
    return Xr

def mrirecon_zf(X, Y, para, pdf):
    Xr=np.zeros_like(X)
    for i in range(X.shape[0]):  
        Xr[i,:,:]=abs(Atmx(Y[i,:,:]/pdf, para.mask)) #/pdf
    
    
    return Xr
    


        
        
    