import sys
sys.path.append('./MRI_Operators')
sys.path.append('./CT_Operators')
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class mnet(nn.Module):
    def __init__(self, in_channels, n_features, use_dropout):
        super(mnet, self).__init__()
        self.conv_enc_mv1 = nn.Conv2d(in_channels, n_features, 3, 1, 1, 1).cuda()
        self.relu_enc_mv1 = nn.ReLU().cuda()
        self.conv_enc_mv2 = nn.Conv2d(n_features, n_features*2, 3, 1, 1, 1).cuda() #8->16 +
        self.relu_enc_mv2 = nn.ReLU().cuda()#
        self.conv_enc_mv3 = nn.Conv2d(n_features*2, n_features*2, 3, 1, 1, 1).cuda() #16->16 +
        self.relu_enc_mv3 = nn.ReLU().cuda()        
        self.conv_enc_tg1 = nn.Conv2d(in_channels, n_features, 3, 1, 1, 1).cuda()
        self.relu_enc_tg1 = nn.ReLU().cuda()
        self.conv_enc_tg2 = nn.Conv2d(n_features, n_features*2, 3, 1, 1, 1).cuda() #8->16 +
        self.relu_enc_tg2 = nn.ReLU().cuda()
        self.conv_enc_tg3 = nn.Conv2d(n_features*2, n_features*2, 3, 1, 1, 1).cuda() #16->16 +
        self.relu_enc_tg3 = nn.ReLU().cuda()
        
        self.conv_enc_m1 = nn.Conv2d(n_features*4, n_features*4, 3, 1, 1, 1).cuda() #32->32
        self.relu_enc_m1 = nn.ReLU().cuda()
        self.conv_enc_m2 = nn.Conv2d(n_features*4, n_features*8, 2, 2, 1, 1).cuda() #32->64 d
        self.relu_enc_m2 = nn.ReLU().cuda()
        self.conv_enc_m3 = nn.Conv2d(n_features*8, n_features*8, 3, 1, 1, 1).cuda() #64->64 +
        self.relu_enc_m3 = nn.ReLU().cuda()
        self.conv_enc_m4 = nn.Conv2d(n_features*8, n_features*8, 3, 1, 1, 1).cuda() #64->64 +
        self.relu_enc_m4 = nn.ReLU().cuda()
        self.conv_enc_m5 = nn.Conv2d(n_features*8, n_features*16, 2, 2, 1, 1).cuda() #64->128 d
        self.relu_enc_m5 = nn.ReLU().cuda()
        self.conv_enc_m6 = nn.Conv2d(n_features*16, n_features*16, 3, 1, 1, 1).cuda() #128->128 +	
        self.relu_enc_m6 = nn.ReLU().cuda()        
        self.conv_enc_m7 = nn.Conv2d(n_features*16, n_features*16, 3, 1, 1, 1).cuda() #128->128 +
        self.relu_enc_m7 = nn.ReLU().cuda()
        self.conv_enc_m8 = nn.Conv2d(n_features*16, n_features*32, 2, 2, 1, 1).cuda() #128->256 d
        self.relu_enc_m8 = nn.ReLU().cuda()
        self.conv_enc_m9 = nn.Conv2d(n_features*32, n_features*32, 3, 1, 1, 1).cuda() #256->256 +
        self.relu_enc_m9 = nn.ReLU().cuda()
        
        self.conv_dec_m9 = nn.ConvTranspose2d(n_features*32, n_features*16, 2, 2, 1, 1).cuda() #256->128 u
        self.relu_dec_m9 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()        
        self.conv_dec_m8 = nn.Conv2d(n_features*32, n_features*16, 3, 1, 1, 1).cuda() #256->128 
        self.relu_dec_m8 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()
        self.conv_dec_m7 = nn.Conv2d(n_features*16, n_features*16, 3, 1, 1, 1).cuda() #128->128 +
        self.relu_dec_m7 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()
        self.conv_dec_m6 = nn.ConvTranspose2d(n_features*16, n_features*8, 2, 2, 1, 1).cuda() #128->64 u
        self.relu_dec_m6 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()        
        self.conv_dec_m5 = nn.Conv2d(n_features*16, n_features*4, 3, 1, 1, 1).cuda() #128->32 u
        self.relu_dec_m5 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()
        self.conv_dec_m4 = nn.Conv2d(n_features*4, n_features*4, 3, 1, 1, 1).cuda() #32->32 +
        self.relu_dec_m4 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()
        self.conv_dec_m3 = nn.ConvTranspose2d(n_features*4, n_features*4, 2, 2, 1, 0).cuda() #32->32  u
        self.relu_dec_m3 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()       
        self.conv_dec_m2 = nn.Conv2d(n_features*8, n_features*2, 3, 1, 1, 1).cuda() #64->16  
        self.conv_dec_m1 = nn.Conv2d(n_features*2, 2, 3, 1, 1, 1).cuda() #16->3                3

                
        self.use_dropout = use_dropout;
        self.dropout = nn.Dropout(0.2).cuda()      
    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input;
    def forward(self, x):
#        [mv, tg] = torch.split(x, 1, 1)
        mv=x[0]
        tg=x[1]
        mv = self.relu_enc_mv1(self.conv_enc_mv1(mv))
        mv = self.relu_enc_mv2(self.conv_enc_mv2(mv))#+mv
        mv = self.relu_enc_mv3(self.conv_enc_mv3(mv))+mv
        tg = self.relu_enc_tg1(self.conv_enc_tg1(tg))
        tg = self.relu_enc_tg2(self.conv_enc_tg2(tg))#+tg
        tg = self.relu_enc_tg3(self.conv_enc_tg3(tg))+tg
        m0 = torch.cat((mv, tg), 1)
        m1 = self.relu_enc_m1(self.conv_enc_m1(m0))
        m2 = self.relu_enc_m2(self.conv_enc_m2(m1))
        m3 = self.relu_enc_m3(self.conv_enc_m3(m2))+m2
        m4 = self.relu_enc_m4(self.conv_enc_m4(m3))+m3
        m5 = self.relu_enc_m5(self.conv_enc_m5(m4))
        m6 = self.relu_enc_m6(self.conv_enc_m6(m5))+m5
        m7 = self.relu_enc_m7(self.conv_enc_m7(m6))+m6
        m8 = self.relu_enc_m8(self.conv_enc_m8(m7))
        m9 = self.relu_enc_m9(self.conv_enc_m9(m8))+m8
        m9_ = torch.cat((self.relu_dec_m9(self.conv_dec_m9(m9)), m7),1)
        m8_ = self.relu_dec_m8(self.conv_dec_m8(m9_))
        m7_ = self.relu_dec_m7(self.conv_dec_m7(m8_))+m8_
        m6_ = torch.cat((self.relu_dec_m6(self.conv_dec_m6(m7_)),m4),1)
        m5_ = self.relu_dec_m5(self.conv_dec_m5(m6_))
        m4_ = self.relu_dec_m4(self.conv_dec_m4(m5_))+m5_        
        m3_ = torch.cat((self.relu_dec_m3(self.conv_dec_m3(m4_)),m1),1)
        m2_ = self.conv_dec_m2(m3_)
        m1_ = self.conv_dec_m1(m2_)


        return m1_.view(-1,2,192,192,1) #3





class fphinet(nn.Module):
    def __init__(self, in_channels, n_features, use_dropout):
        super(fphinet, self).__init__()
        self.conv_enc_mv1 = nn.Conv2d(n_features, in_channels, 3, 1, 1, 1).cuda()
        self.relu_enc_mv1 = nn.ReLU().cuda()
        self.conv_enc_mv2 = nn.Conv2d(2*n_features, n_features, 3, 1, 1, 1).cuda() #16->8 +
        self.relu_enc_mv2 = nn.ReLU().cuda()#
        self.conv_enc_mv3 = nn.Conv2d(n_features*2, n_features*2, 3, 1, 1, 1).cuda() #16->16 +
        self.relu_enc_mv3 = nn.ReLU().cuda()  
        self.conv_enc_mv4 = nn.Conv2d(n_features*4, n_features*2, 3, 1, 1, 1).cuda() #32->16 +
        self.relu_enc_mv4 = nn.ReLU().cuda() 
        
        self.conv_enc_tg1_ = nn.Conv2d(in_channels, n_features, 3, 1, 1, 1).cuda()
        self.relu_enc_tg1_ = nn.ReLU().cuda()
        self.conv_enc_tg2_ = nn.Conv2d(n_features, n_features*2, 3, 1, 1, 1).cuda() #8->16 +
        self.relu_enc_tg2_ = nn.ReLU().cuda()
        self.conv_enc_tg3_ = nn.Conv2d(n_features*2, n_features*2, 3, 1, 1, 1).cuda() #16->16 +
        self.relu_enc_tg3_ = nn.ReLU().cuda()
 
        self.conv_enc_m0 = nn.Conv2d(in_channels*2, n_features*2, 3, 1, 1, 1).cuda() #2->16        3
        self.relu_enc_m0 = nn.ReLU().cuda()       
        self.conv_enc_m1 = nn.Conv2d(n_features*2, n_features*4, 3, 1, 1, 1).cuda() #16->32
        self.relu_enc_m1 = nn.ReLU().cuda()
        self.conv_enc_m2 = nn.Conv2d(n_features*4, n_features*8, 2, 2, 1, 1).cuda() #32->64 d
        self.relu_enc_m2 = nn.ReLU().cuda()
        self.conv_enc_m3 = nn.Conv2d(n_features*8, n_features*8, 3, 1, 1, 1).cuda() #64->64 +
        self.relu_enc_m3 = nn.ReLU().cuda()
        self.conv_enc_m4 = nn.Conv2d(n_features*8, n_features*8, 3, 1, 1, 1).cuda() #64->64 +
        self.relu_enc_m4 = nn.ReLU().cuda()
        self.conv_enc_m5 = nn.Conv2d(n_features*8, n_features*16, 2, 2, 1, 1).cuda() #64->128 d
        self.relu_enc_m5 = nn.ReLU().cuda()
        self.conv_enc_m6 = nn.Conv2d(n_features*16, n_features*16, 3, 1, 1, 1).cuda() #128->128 +	
        self.relu_enc_m6 = nn.ReLU().cuda()        
        self.conv_enc_m7 = nn.Conv2d(n_features*16, n_features*16, 3, 1, 1, 1).cuda() #128->128 +
        self.relu_enc_m7 = nn.ReLU().cuda()
        self.conv_enc_m8 = nn.Conv2d(n_features*16, n_features*32, 2, 2, 1, 1).cuda() #128->256 d
        self.relu_enc_m8 = nn.ReLU().cuda()
        self.conv_enc_m9 = nn.Conv2d(n_features*32, n_features*32, 3, 1, 1, 1).cuda() #256->256 +
        self.relu_enc_m9 = nn.ReLU().cuda()
        
        self.conv_dec_m9 = nn.ConvTranspose2d(n_features*32*2, n_features*16*2, 2, 2, 1, 1).cuda() #256->128 u
        self.relu_dec_m9 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()        
        self.conv_dec_m8 = nn.Conv2d(n_features*32*2, n_features*16*2, 3, 1, 1, 1).cuda() #256->128 
        self.relu_dec_m8 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()
        self.conv_dec_m7 = nn.Conv2d(n_features*16*2, n_features*16*2, 3, 1, 1, 1).cuda() #128->128 +
        self.relu_dec_m7 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()
        self.conv_dec_m6 = nn.ConvTranspose2d(n_features*16*2, n_features*8*2, 2, 2, 1, 1).cuda() #128->64 u
        self.relu_dec_m6 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()        
        self.conv_dec_m5 = nn.Conv2d(n_features*16*2, n_features*4*2, 3, 1, 1, 1).cuda() #128->32 u
        self.relu_dec_m5 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()
        self.conv_dec_m4 = nn.Conv2d(n_features*4*2, n_features*4*2, 3, 1, 1, 1).cuda() #32->32 +
        self.relu_dec_m4 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()
        self.conv_dec_m3 = nn.ConvTranspose2d(n_features*4*2, n_features*4*2, 2, 2, 1, 0).cuda() #32->32  u
        self.relu_dec_m3 = nn.LeakyReLU(negative_slope=0.01, inplace=False).cuda()        
        self.conv_dec_m2 = nn.Conv2d(n_features*8*2, n_features*2*2, 3, 1, 1, 1).cuda() #64->16  

        self.conv_enc_tg1 = nn.Conv2d(n_features*2, n_features*4, 3, 1, 1, 1).cuda() #16->32
        self.relu_enc_tg1 = nn.ReLU().cuda()
        self.conv_enc_tg2 = nn.Conv2d(n_features*4, n_features*8, 2, 2, 1, 1).cuda() #32->64 d
        self.relu_enc_tg2 = nn.ReLU().cuda()
        self.conv_enc_tg3 = nn.Conv2d(n_features*8, n_features*8, 3, 1, 1, 1).cuda() #64->64 +
        self.relu_enc_tg3 = nn.ReLU().cuda()
        self.conv_enc_tg4 = nn.Conv2d(n_features*8, n_features*8, 3, 1, 1, 1).cuda() #64->64 +
        self.relu_enc_tg4 = nn.ReLU().cuda()
        self.conv_enc_tg5 = nn.Conv2d(n_features*8, n_features*16, 2, 2, 1, 1).cuda() #64->128 d
        self.relu_enc_tg5 = nn.ReLU().cuda()
        self.conv_enc_tg6 = nn.Conv2d(n_features*16, n_features*16, 3, 1, 1, 1).cuda() #128->128 +	
        self.relu_enc_tg6 = nn.ReLU().cuda()        
        self.conv_enc_tg7 = nn.Conv2d(n_features*16, n_features*16, 3, 1, 1, 1).cuda() #128->128 +
        self.relu_enc_tg7 = nn.ReLU().cuda()
        self.conv_enc_tg8 = nn.Conv2d(n_features*16, n_features*32, 2, 2, 1, 1).cuda() #128->256 d
        self.relu_enc_tg8 = nn.ReLU().cuda()
        self.conv_enc_tg9 = nn.Conv2d(n_features*32, n_features*32, 3, 1, 1, 1).cuda() #256->256 +
        self.relu_enc_tg9 = nn.ReLU().cuda()        


                
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.2).cuda()      
    def apply_dropout(self, input):
        if self.use_dropout:
            return self.dropout(input)
        else:
            return input;
    def forward(self, x):
#        [mv, tg] = torch.split(x, 1, 1)
        m0=x[0]
        tg=x[1]
        m0=m0.view(-1,2,192,192)#3
        tg = self.relu_enc_tg1_(self.conv_enc_tg1_(tg))
        tg = self.relu_enc_tg2_(self.conv_enc_tg2_(tg))#+tg
        tg = self.relu_enc_tg3_(self.conv_enc_tg3_(tg))+tg
        tg1 = self.relu_enc_tg1(self.conv_enc_tg1(tg))
        tg2 = self.relu_enc_tg2(self.conv_enc_tg2(tg1))
        tg3 = self.relu_enc_tg3(self.conv_enc_tg3(tg2))+tg2
        tg4 = self.relu_enc_tg4(self.conv_enc_tg4(tg3))+tg3
        tg5 = self.relu_enc_tg5(self.conv_enc_tg5(tg4))
        tg6 = self.relu_enc_tg6(self.conv_enc_tg6(tg5))+tg5
        tg7 = self.relu_enc_tg7(self.conv_enc_tg7(tg6))+tg6
        tg8 = self.relu_enc_tg8(self.conv_enc_tg8(tg7))
        tg9 = self.relu_enc_tg9(self.conv_enc_m9(tg8))+tg8
        

        m00 = self.relu_enc_m0(self.conv_enc_m0(m0))      
        m1 = self.relu_enc_m1(self.conv_enc_m1(m00))
        m2 = self.relu_enc_m2(self.conv_enc_m2(m1))
        m3 = self.relu_enc_m3(self.conv_enc_m3(m2))+m2
        m4 = self.relu_enc_m4(self.conv_enc_m4(m3))+m3
        m5 = self.relu_enc_m5(self.conv_enc_m5(m4))
        m6 = self.relu_enc_m6(self.conv_enc_m6(m5))+m5
        m7 = self.relu_enc_m7(self.conv_enc_m7(m6))+m6
        m8 = self.relu_enc_m8(self.conv_enc_m8(m7))
        m9 = self.relu_enc_m9(self.conv_enc_m9(m8))+m8
        m9c = torch.cat((m9, tg9), 1)        
        
        
        m9_ = torch.cat((self.relu_dec_m9(self.conv_dec_m9(m9c)), m7,tg7),1)
        m8_ = self.relu_dec_m8(self.conv_dec_m8(m9_))
        m7_ = self.relu_dec_m7(self.conv_dec_m7(m8_))+m8_
        m6_ = torch.cat((self.relu_dec_m6(self.conv_dec_m6(m7_)),m4,tg4),1)
        m5_ = self.relu_dec_m5(self.conv_dec_m5(m6_))
        m4_ = self.relu_dec_m4(self.conv_dec_m4(m5_))+m5_        
        m3_ = torch.cat((self.relu_dec_m3(self.conv_dec_m3(m4_)),m1,tg1),1)
        m2_ = self.conv_dec_m2(m3_)
        
        
        mv = self.relu_enc_mv4(self.conv_enc_mv4(m2_))
        mv = self.relu_enc_mv3(self.conv_enc_mv3(mv))+mv
        mv = self.relu_enc_mv2(self.conv_enc_mv2(mv))
        mv = self.conv_enc_mv1(mv)

        return mv
    


class priornet(nn.Module):
    def __init__(self, in_channels, n_features, use_dropout):
        super(priornet, self).__init__()
        self.mnet_ = mnet(in_channels, n_features, 0)
        self.fphinet_ = fphinet(in_channels, n_features, 0)
    def forward(self, x):
        m0=self.mnet_(x)
        mv=self.fphinet_((m0,x[1]))

        return (m0,mv)         
    



class inv_cg_Function(torch.autograd.Function):

    @staticmethod
    def forward(self, im,Y,rho,M):
        imshape=im.shape
#        rho = 1.0/(1.0+torch.exp(-rho))
        M.set_rho(rho.data.cpu().numpy())
        self.M = M
        self.imshape=imshape
#        print(im.size())
        mv= im.data.cpu().view(-1,imshape[2],imshape[3]).numpy() #.view(-1,imshape[2],imshape[3])
        AtY = Y.data.cpu().numpy()
        X=self.M.recon_cg_batch(AtY, mv)
        X=torch.from_numpy(X).view(-1,1,imshape[2],imshape[3]).cuda() 
#
        self.save_for_backward(im,Y,X)
        return X 

    @staticmethod
    def backward(self, grad_output):
        mv0,Y,X, = self.saved_tensors
#        mv= mv0.data.cpu().view(-1,self.imshape[2],self.imshape[3]).numpy()
        grad_input = grad_output.clone()
        grad_input_np = grad_input.data.cpu().view(-1,self.imshape[2],self.imshape[3]).numpy()*self.M.rho
        X=self.M.recon_cg_batch(np.zeros_like(Y.cpu().numpy()), grad_input_np)
        X=torch.from_numpy(X).view(-1,1,self.imshape[2],self.imshape[3]).cuda()
        grad_v = Variable(X).cuda()
#        grad_y = Variable(torch.cat((X/self.M.rho,X/self.M.rho),1)).cuda() # no need to compute gra_out2,  save computation time
#        T = (mv0 - X)*grad_input
        T = (mv0 - X)*grad_v/torch.Tensor(self.M.rho).cuda()
        grad_rho = torch.sum(T, (2, 3))
#        grad_rho = 1.0/(1.0+torch.exp(-grad_rho))*(1.0-1.0/(1.0+torch.exp(-grad_rho)))
        return grad_v, None, grad_rho , None
#        return grad_out



class inv_cg(nn.Module):
    def __init__(self, M):
        super(inv_cg, self).__init__()
        self.M = M
#        self.rho = nn.Parameter(torch.Tensor(1))
#        self.rho.data.fill_(M.rho)#-np.log(1.0/M.rho-1.0))
        self.sigmoid  = nn.Sigmoid().cuda()
        self.linear = nn.Linear(1, 1, bias=False).cuda()        
        self.rho = Variable(M.rho*torch.ones(1).cuda())
    def forward(self, im, y):
        rho_l = 0.8*self.sigmoid(self.linear(self.rho))
        output = inv_cg_Function.apply(im, y, rho_l, self.M)
        return output



class reconregnet(nn.Module):
    def __init__(self, in_channels, n_features, use_dropout, M):
        super(reconregnet, self).__init__()
        self.priornetblk_1 = priornet(in_channels, n_features, use_dropout)
        self.priornetblk_2 = priornet(in_channels, n_features, use_dropout)
        self.priornetblk_3 = priornet(in_channels, n_features, use_dropout)
        self.invblk_1 = inv_cg(M[0])
        self.invblk_2 = inv_cg(M[1])
        self.invblk_3 = inv_cg(M[2])

    def forward(self, x):

        (mvrc0,tg,Y) = x
        t0=mvrc0
        (m0p1,mvp1) = self.priornetblk_1((t0,tg))
        ti1=2*mvp1-t0
        mvrc1 = self.invblk_1(ti1,Y)
        t1 = t0-mvp1+mvrc1
        (m0p2,mvp2) = self.priornetblk_2((t1,tg))
        ti2=2*mvp2-t1	
        mvrc2 = self.invblk_2(ti2,Y)
        t2 = t1-mvp2+mvrc2
        (m0p3,mvp3) = self.priornetblk_3((t2,tg))  
        ti3=2*mvp3-t1	
        mvrc3 = self.invblk_3(ti3,Y)

        return (m0p1, mvp1, m0p2, mvp2, m0p3, mvp3, mvrc1, mvrc2, mvrc3)




