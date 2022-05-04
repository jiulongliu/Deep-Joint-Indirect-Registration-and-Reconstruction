# add LDDMM shooting code into path
import sys
#sys.path.append('../vectormomentum/Code/Python')
#sys.path.append('../library')
#from subprocess import call
import argparse
#import os.path
import gc
import torchfile
#Add deep learning related libraries
#from collections import Counter
import torch
import torch.utils.data
#from torch.utils.serialization import load_lua
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.autograd import Variable
import inversion_n_registration_network
#import util
import numpy as np
import scipy.io
#import mrirecon3d
import pymrirecon 
#Add LDDMM registration related libraries
# library for importing LDDMM formulation configs
import yaml
# others
#import logging
#import copy
#import math

#parse command line input
parser = argparse.ArgumentParser(description='Deformation predicting given set of moving and target images.')

##required parameters
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('--moving-image-dataset', nargs='+', required=True, metavar=('m1', 'm2, m3...'),
                           help='List of moving images datasets stored in .pth.tar format (or .t7 format for the old experiments in the Neuroimage paper). File names are seperated by space.')
requiredNamed.add_argument('--target-image-dataset', nargs='+', required=True, metavar=('t1', 't2, t3...'),
                           help='List of target images datasets stored in .pth.tar format (or .t7 format for the old experiments in the Neuroimage paper). File names are seperated by space.')
requiredNamed.add_argument('--deformation-parameter', nargs='+', required=True, metavar=('o1', 'o2, o3...'),
                           help='List of target deformation parameter files to predict to, stored in .pth.tar format (or .t7 format for the old experiments in the Neuroimage paper). File names are seperated by space.')
requiredNamed.add_argument('--deformation-setting-file', required=True, metavar=('yaml_file'),
                           help='yaml file that contains LDDMM registration settings.')
requiredNamed.add_argument('--output-name', required=True, metavar=('file_name'),
                           help='output directory + name of the network parameters, in .pth.tar format.')
requiredNamed.add_argument('--mask', required=True, metavar=('file_name'),
                           help='mask input directory + name of the network parameters, in .mat format.')

##optional parameters
parser.add_argument('--image-size-nx', type=int, default=192, metavar='N',
                    help='image height')
parser.add_argument('--image-size-ny', type=int, default=192, metavar='N',
                    help='image width')
parser.add_argument('--features', type=int, default=8, metavar='N',
                    help='number of output features for the first layer of the deep network (default: 64)')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for prediction network (default: 64)')
parser.add_argument('--patch-size', type=int, default=31, metavar='N',
                    help='patch size to extract patches (default: 15)')
parser.add_argument('--stride', type=int, default=9, metavar='N',
                    help='sliding window stride to extract patches for training (default: 14)')
parser.add_argument('--epochs', type=int, default=4000, metavar='N',
                    help='number of epochs to train the network (default: 10)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='N',
                    help='learning rate for the adam optimization algorithm (default: 0.0001)')
parser.add_argument('--use-dropout', action='store_true', default=False,
                    help='Use dropout to train the probablistic version of the network')
parser.add_argument('--n-GPU', type=int, default=1, metavar='N',
                    help='number of GPUs used for training.')
parser.add_argument('--continue-from-parameter', metavar=('parameter_name'),
                           help='file directory+name of the existing parameter if want to start')
args = parser.parse_args()

# finish command line input


# check validity of input arguments from command line
def check_args(args):
    # number of input images/output prefix consistency check
    n_moving_images = len(args.moving_image_dataset)
    n_target_images = len(args.target_image_dataset)
    n_deformation_parameter = len(args.deformation_parameter)
    if (n_moving_images != n_target_images):
        print('The number of moving image datasets is not consistent with the number of target image datasets!')
        sys.exit(1)
    elif (n_moving_images != n_deformation_parameter ):
        print('The number of moving image datasets is not consistent with the number of deformation parameter datasets!')
        sys.exit(1)

    # number of GPU check (positive integers)
    if (args.n_GPU <= 0):
        print('Number of GPUs must be positive!')
        sys.exit(1)
#enddef

def read_spec(args):
    stream = open(args.deformation_setting_file, 'r')
    usercfg = yaml.load(stream, Loader = yaml.Loader)
    return usercfg
#enddef




#class DLossFunc(nn.Module):
#    def __init__(self,  alpha=1000, beta=1.0,gamma=1000.0,delta=1.0, lossit=1):
#        super(DLossFunc, self).__init__()
##        self.t1 = t1
##        self.t2 = t2
#        self.alpha = alpha
#        self.beta = beta
#        self.gamma = gamma
#        self.delta = delta 
#        self.lossit = lossit         
#        return
#
#    def forward(self, m0p1, mvp1, m0p2, mvp2, m0p3, mvp3, m0, mv):    
##        print('m0gd',m0.shape)
#        d1=func.l1_loss(m0p1, m0, size_average=None, reduce=None, reduction='sum') 
##        print('d1',d1)
#        d2=func.mse_loss(mvp1,mv, size_average=None, reduce=None, reduction='sum') 
#        d3=func.l1_loss(m0p2, m0,  size_average=None, reduce=None, reduction='sum') 
#        d4=func.mse_loss( mvp2, mv, size_average=None, reduce=None, reduction='sum')
#        d5=func.l1_loss(m0p3, m0,  size_average=None, reduce=None, reduction='sum') 
#        d6=func.mse_loss( mvp3, mv, size_average=None, reduce=None, reduction='sum')
##        loss = torch.mean(d1*self.alpha)#+d2*self.beta+d3*self.gamma+d4*self.delta)
#        if self.lossit==1:
#            return d1*700+d2*0.10
#        elif self.lossit==2:
#            return d3*800+d4*0.12
#        elif self.lossit==3:
#            return d5*900+d6*0.15
#        elif self.lossit==4:
#            return d1*700+d2*0.10+d3*800+d4*0.12+d5*900+d6*0.15
#        else:
#            return d1*self.alpha+d2*self.beta+d3*self.gamma+d4*self.delta
##        return d1*700+d2*0.10+d3*800+d4*0.12+d5*900+d6*0.15


class DLossFunc(nn.Module):
    def __init__(self,  a1=700, a2=800, a3=900, b1=0.50, b2=0.60, b3=0.75, lossit=1):
        super(DLossFunc, self).__init__()
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3   
        self.lossit = lossit        
        return
    def forward(self, m0p1, mvp1, m0p2, mvp2, m0p3, mvp3, m0, mv):    
        d1=func.mse_loss(m0p1, m0, size_average=None, reduce=None, reduction='sum') 
        e1=func.mse_loss(mvp1, mv, size_average=None, reduce=None, reduction='sum') 
        d2=func.mse_loss(m0p2, m0, size_average=None, reduce=None, reduction='sum') 
        e2=func.mse_loss(mvp2, mv, size_average=None, reduce=None, reduction='sum')
        d3=func.mse_loss(m0p3, m0, size_average=None, reduce=None, reduction='sum') 
        e3=func.mse_loss(mvp3, mv, size_average=None, reduce=None, reduction='sum')
        if self.lossit==1:
            return d1*self.a1+e1*self.b1
        elif self.lossit==2:
            return d2*self.a2+e2*self.b2
        elif self.lossit==3:
            return d3*self.a3+e3*self.b3
        elif self.lossit==4:
            return d1*self.a1+e1*self.b1+d2*self.a2+e2*self.b2+d3*self.a3+e3*self.b3
        else:
            return d1*self.a1+e1*self.b1+d2*self.a2+e2*self.b2+d3*self.a3+e3*self.b3
#        return d1*700+d2*0.10+d3*800+d4*0.12+d5*900+d6*0.15





def create_net(args):
    mask = scipy.io.loadmat(args.mask)['k']
#    print(args.mask,mask)
    mask_size = mask.shape
#    print(mask_size)
    if (mask_size[0]!=args.image_size_nx):
        print('mask width not correct')
        sys.exit(1)
    if (mask_size[1]!=args.image_size_ny):
        print('mask height not correct')
        sys.exit(1)
    maski=np.fft.ifftshift(mask)
    M=[]
    for i in range(0, 3):
        Mi=pymrirecon.mriroplib(mask=maski,N=args.image_size_nx, mu=0.0,rho=0.4,lam=0.08, N_iter=0, CG_iter=40,CG_tol=1e-7,Min_iter=10)
        M.append(Mi)
    net_single = inversion_n_registration_network.reconregnet(1,args.features, args.use_dropout, M).cuda()
    if (args.continue_from_parameter != None):
        print('Loading existing parameter file!')
        config = torch.load(args.continue_from_parameter)
        net_single.load_state_dict(config['state_dict'])

    if (args.n_GPU > 1) :
        device_ids=range(0, args.n_GPU)
        net = torch.nn.DataParallel(net_single, device_ids=device_ids).cuda()
    else:
        net = net_single

    net.train()
    return net


def train_cur_data(cur_epoch, datapart, moving_file, target_file, parameter, output_name, net, criterion, optimizer, registration_spec, args):
    old_experiments = False
    if moving_file[-3:] == '.t7' :
        old_experiments = True
        #only for old data used in the Neuroimage paper. Do not use .t7 format for new data and new experiments.
        moving_appear_trainset = torchfile.load(moving_file).float()
        target_appear_trainset = torchfile.load(target_file).float()
        train_m0 = torchfile.load(parameter).float()
    else :
        moving_appear_trainset = torch.load(moving_file).float()
        target_appear_trainset = torch.load(target_file).float()
        train_m0 = torch.load(parameter).float()
    dataset_size = moving_appear_trainset.size()    
    if (dataset_size[1]!=args.image_size_nx):
        print('images width not correct')
        sys.exit(1)
    if (dataset_size[2]!=args.image_size_ny):
        print('images height not correct')
        sys.exit(1)
    
#    print(dataset_size)
    N = dataset_size[0] / args.batch_size;    

    noiselevel = 0.1
#    M = net.invblk_1.M
    mask = scipy.io.loadmat(args.mask)['k']
    maski=np.fft.ifftshift(mask)
    M=pymrirecon.mriroplib(mask=maski,N=args.image_size_nx,mu=0.4,rho=0.0,lam=0.08, N_iter=20, CG_iter=20,CG_tol=1e-9,Min_iter=10)  
    Y_part = M.genksdata(moving_appear_trainset.view(-1,args.image_size_nx,args.image_size_ny).numpy(), noiselevel = noiselevel)  
#    M.__init__(mu=0.4,rho=0.0,lam=0.08, N_iter=20, CG_iter=20,CG_tol=1e-9,Min_iter=10) 
    mvrc0_batch_np = M.recon_admm_batch(Y_part, np.zeros_like(moving_appear_trainset.view(-1,args.image_size_nx,args.image_size_ny).numpy()))
#    M.__init__(mu=0.0,rho=0.4,lam=0.08, N_iter=0, CG_iter=40,CG_tol=1e-7,Min_iter=10)
#    scipy.io.savemat("output/train/imtrain_mvrc0_"+args.mask[-18:-4]+"_ep_"+str(cur_epoch+1)+".mat", {'mvrc0_batch_np': mvrc0_batch_np,
#                                                 'Y_part':Y_part,
#                                                 'moving_appear_trainset':moving_appear_trainset.view(-1,args.image_size_nx,args.image_size_ny).numpy()})#enddef    
    mvrc0_part = torch.from_numpy(mvrc0_batch_np)    
    Y_part = torch.from_numpy(Y_part)
    torch_dataset = torch.utils.data.TensorDataset(moving_appear_trainset,target_appear_trainset,train_m0[:,0:2,:,:,:],Y_part, mvrc0_part)        
    train_loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)              #
    for i, (mv_batch,tg_batch,output_batch,Y_batch,mvrc0_batch) in enumerate(train_loader):
        mvrc0_batch_variable = Variable(mvrc0_batch.view(-1,1,args.image_size_nx,args.image_size_ny)).cuda()
        Y_batch_variable = Variable(Y_batch.view(-1,2,args.image_size_nx,args.image_size_ny)).cuda()
        mv_batch_variable = Variable(mv_batch.view(-1,1,args.image_size_nx,args.image_size_ny)).cuda()
        tg_batch_variable = Variable(tg_batch.view(-1,1,args.image_size_nx,args.image_size_ny)).cuda()
        input_batch_variable = (mvrc0_batch_variable,tg_batch_variable,Y_batch_variable) 
        output_batch_variable = Variable(output_batch).cuda()
     
        optimizer.zero_grad()
        (m0p1_batch_variable, mvp1_batch_variable, m0p2_batch_variable, mvp2_batch_variable,m0p3_batch_variable, mvp3_batch_variable, mvrc1_batch_variable, mvrc2_batch_variable, mvrc3_batch_variable) = net(input_batch_variable) #(m0p1,mvp1,m0p2,mvp2,mvrc2)
        loss = criterion(m0p1_batch_variable, mvp1_batch_variable, m0p2_batch_variable, mvp2_batch_variable, m0p3_batch_variable, mvp3_batch_variable, output_batch_variable,mv_batch_variable) #[:][:,0:1,:,:,:]
        loss.backward()
        loss_value = loss.data.item()
        optimizer.step()
        
        print('====> Epoch: {}, datapart: {}, iter: {}/{}, mri_recon+reg loss: {:.10f}, rho1: {}, rho2: {}, rho3: {}'.format(
            cur_epoch+1, datapart+1, i, N, loss_value/args.batch_size, net.invblk_1.M.rho, net.invblk_2.M.rho, net.invblk_3.M.rho))
        if i % 100 == 0 or i == N-1:
            if args.n_GPU > 1:
                cur_state_dict = net.module.state_dict()
            else:
                cur_state_dict = net.state_dict()                        
            modal_name = output_name            
            model_info = {
                'patch_size' : args.patch_size,
                'network_feature' : args.features,
                'state_dict': cur_state_dict,
                'deformation_params': registration_spec,
                'last_epoch': cur_epoch
            }
            if old_experiments :
                model_info['matlab_t7'] = True
            
           
            torch.save(model_info, modal_name)
            if (cur_epoch+1)%50==0:
                torch.save(model_info, "output/train/prediction_mri_recon_indreg_"+args.mask[-18:-4]+"_ep_"+str(cur_epoch+1)+".pth.tar")
                scipy.io.savemat("output/train/imtrain_training_mri_"+args.mask[-18:-4]+"_ep_"+str(cur_epoch+1)+".mat", {'m0p1': m0p1_batch_variable.detach().view(-1,2,args.image_size_nx, args.image_size_ny).cpu().numpy(),
                                                 'm0p2':m0p2_batch_variable.detach().view(-1,2,args.image_size_nx,args.image_size_ny).cpu().numpy(),  
                                                 'm0p3':m0p3_batch_variable.detach().view(-1,2,args.image_size_nx,args.image_size_ny).cpu().numpy(),                                              
                                                 'm0gd':output_batch_variable.detach().view(-1,2,args.image_size_nx,args.image_size_ny).cpu().numpy(),
                                                 'mvp1':mvp1_batch_variable.detach().view(-1,1,args.image_size_nx, args.image_size_ny).cpu().numpy(),
                                                 'mvp2':mvp2_batch_variable.detach().view(-1,1,args.image_size_nx, args.image_size_ny).cpu().numpy(),
                                                 'mvp3':mvp3_batch_variable.detach().view(-1,1,args.image_size_nx, args.image_size_ny).cpu().numpy(),                                             
                                                 'mvrc1':mvrc1_batch_variable.detach().view(-1,1,args.image_size_nx, args.image_size_ny).cpu().numpy(),
                                                 'mvrc2':mvrc2_batch_variable.detach().view(-1,1,args.image_size_nx, args.image_size_ny).cpu().numpy(),
                                                 'mvrc3':mvrc3_batch_variable.detach().view(-1,1,args.image_size_nx, args.image_size_ny).cpu().numpy(),                                             
                                                 'mvrc0':mvrc0_batch_variable.detach().view(-1,1,args.image_size_nx, args.image_size_ny).cpu().numpy(),                                                
                                                 'mvgd': mv_batch_variable.detach().view(-1,1,args.image_size_nx, args.image_size_ny).cpu().numpy(),
                                                 'tggd': tg_batch_variable.detach().view(-1,1,args.image_size_nx, args.image_size_ny).cpu().numpy(),})#enddef    



def train_network(args, registration_spec):
    net = create_net(args)
    net.train()
#    criterion = nn.L1Loss(False).cuda()
    criterion = DLossFunc(a1=700, a2=800, a3=900, b1=0.50, b2=0.60, b3=0.75, lossit=1)#.cuda()
#    criterion = nn.L1Loss(False).cuda()
    optimizer = optim.Adam(net.parameters(), args.learning_rate)
    if (args.continue_from_parameter != None):
        print('Continue training from last train!')
        config = torch.load(args.continue_from_parameter)
        last_epoch = config['last_epoch']
    else:
        last_epoch = -1
        
    if last_epoch <600 and last_epoch >300:   
        criterion.__init__(a1=700, a2=800, a3=900, b1=0.50, b2=0.60, b3=0.75, lossit=2)
        ct = 0            
        for child in net.children():
            ct += 1
            if ct < 2:
                 for param in child.parameters():
                    param.requires_grad = False   
    elif last_epoch <900 and last_epoch >600:
        criterion.__init__(a1=700, a2=800, a3=900, b1=0.50, b2=0.60, b3=0.75, lossit=3)
        ct = 0            
        for child in net.children():
            ct += 1
            if ct < 3:
                 for param in child.parameters():
                    param.requires_grad = False           
    elif last_epoch >900:
        criterion.__init__(a1=700, a2=800, a3=900, b1=0.50, b2=0.60, b3=0.75, lossit=4)            
        ct = 0            
        for child in net.children():
            ct += 1
            if ct < 4:
                 for param in child.parameters():
                    param.requires_grad = True       
        
        
    for cur_epoch in range(last_epoch+1, args.epochs) :            
        if cur_epoch ==300:
            criterion.__init__(a1=700, a2=800, a3=900, b1=0.50, b2=0.60, b3=0.75, lossit=2)
            ct = 0            
            for child in net.children():
                ct += 1
                if ct < 2:
                     for param in child.parameters():
                        param.requires_grad = False   
#            print(list(list(net.children())[0].named_parameters()))
#            print(len(list(net.children())))
#            print(len(list(list(net.children())[0].named_parameters())))
#            print("------------------------------\n")
#            print(list(list(net.children())[1].named_parameters()))
#            print("------------------------------\n")
#            print(list(list(net.children())[2].named_parameters()))
#            sys.exit(1)
        if cur_epoch ==600:
            criterion.__init__(a1=700, a2=800, a3=900, b1=0.50, b2=0.60, b3=0.75, lossit=3)   
            ct = 0   
            for child in net.children():
                ct += 1
                if ct < 3:
                     for param in child.parameters():
                        param.requires_grad = False  
        if cur_epoch ==900:
            criterion.__init__(a1=700, a2=800, a3=900, b1=0.50, b2=0.60, b3=0.75, lossit=4)   
            ct = 0   
            for child in net.children():
                ct += 1
                if ct < 4:
                     for param in child.parameters():
                        param.requires_grad = True             
        for datapart in range(0, len(args.moving_image_dataset)) :
            train_cur_data(
                cur_epoch, 
                datapart,
                args.moving_image_dataset[datapart], 
                args.target_image_dataset[datapart], 
                args.deformation_parameter[datapart], 
                args.output_name,
                net, 
                criterion, 
                optimizer,
                registration_spec,
                args
            )
            gc.collect()
            


if __name__ == '__main__':
    check_args(args);
    registration_spec = read_spec(args)
    train_network(args, registration_spec)
