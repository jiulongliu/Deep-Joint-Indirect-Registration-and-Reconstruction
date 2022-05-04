# add LDDMM shooting code into path
import sys
sys.path.append('3rd_party_software/vectormomentum/Code/Python')
sys.path.append('MRI_Operators')
#sys.path.append('3rd_party_software/')
sys.path.append('3rd_party_software/geodesic_shooting/')
from subprocess import call
import argparse
import os.path


#Add deep learning related libraries
from collections import Counter
import torch
import h5py
from torch.autograd import Variable
import inversion_n_registration_network
import util
import numpy as np
from skimage import exposure
import pymrirecon 
#Add LDDMM registration related libraries
# pyca modules
import PyCA.Core as ca
import PyCA.Common as common
#import PyCA.Display as display
# vector momentum modules
# others
import logging
import copy
import math

import registration_methods

import scipy.io

#parse command line input
parser = argparse.ArgumentParser(description='Reconstruction and Deformation prediction given set of moving and target images.')

requiredNamed = parser.add_argument_group('required named arguments')

requiredNamed.add_argument('--moving-image', nargs='+', required=True, metavar=('m1', 'm2, m3...'),
                           help='List of moving images, seperated by space.')
requiredNamed.add_argument('--target-image', nargs='+', required=True, metavar=('t1', 't2, t3...'),
                           help='List of target images, seperated by space.')
requiredNamed.add_argument('--output-prefix', nargs='+', required=True, metavar=('o1', 'o2, o3...'),
                           help='List of registration output prefixes for every moving/target image pair, seperated by space. Preferred to be a directory (e.g. /some_path/output_dir/)')
requiredNamed.add_argument('--mask', required=True, metavar=('file_name'),
                           help='mask input directory + name of the network parameters, in .mat format.')

parser.add_argument('--image-size-nx', type=int, default=192, metavar='N',
                    help='image height')
parser.add_argument('--image-size-ny', type=int, default=192, metavar='N',
                    help='image width')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for prediction network (default: 64)')
parser.add_argument('--features', type=int, default=8, metavar='N',
                    help='number of output features for the first layer of the deep network (default: 64)')
parser.add_argument('--n-GPU', type=int, default=1, metavar='N',
                    help='number of GPUs used for prediction (default: 1). For maximum efficiency please set the batch size divisible by the number of GPUs.')
parser.add_argument('--use-correction', action='store_true', default=False,
                    help='Apply correction network after prediction network. Slower computation time but with potential better registration accuracy.')
parser.add_argument('--use-CPU-for-shooting', action='store_true', default=False,
                    help='Use CPU for geodesic shooting. Slow, but saves GPU memory.')
parser.add_argument('--shoot-steps', type=int, default=0, metavar='N',
                    help='time steps for geodesic shooting. Ignore this option to use the default step size used by the registration model.')
parser.add_argument('--affine-align', action='store_true', default=False,
                    help='Perform affine registration to align moving and target images to ICBM152 atlas space. Require niftireg.')
parser.add_argument('--histeq', action='store_true', default=False,
                    help='Perform histogram equalization to the moving and target images.')
parser.add_argument('--atlas', default="../data/atlas/icbm152.nii",
                    help="Atlas to use for (affine) pre-registration")
parser.add_argument('--prediction-parameter', default='output/prediction_mri_005_1s.pth.tar',
                    help="network parameters for the inverse and registration network")
#parser.add_argument('--correction-parameter', default='../../network_configs/OASIS_correct.pth.tar',
#                    help="network parameters for the correction network")
args = parser.parse_args()


# check validity of input arguments from command line
def check_args(args):
    # number of input images/output prefix consistency check
    n_moving_images = len(args.moving_image)
    n_target_images = len(args.target_image)
    n_output_prefix = len(args.output_prefix)
    if (n_moving_images != n_target_images):
        print('The number of moving images is not consistent with the number of target images!')
        sys.exit(1)
    elif (n_moving_images != n_output_prefix ):
        print('The number of output prefix is not consistent with the number of input images!')
        sys.exit(1)

    # number of GPU check (positive integers)
    if (args.n_GPU <= 0):
        print('Number of GPUs must be positive!')
        sys.exit(1)

    # geodesic shooting step check (positive integers)
    if (args.shoot_steps < 0):
        print('Shooting steps (--shoot-steps) is negative. Using model default step.')
#enddef


def create_net(args, network_config):
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
    net_single = inversion_n_registration_network.reconregnet(1,args.features, 0, M).cuda()
    net_single.load_state_dict(network_config['state_dict'])

    if (args.n_GPU > 1) :
        device_ids=range(0, args.n_GPU)
        net = torch.nn.DataParallel(net_single, device_ids=device_ids).cuda()
    else:
        net = net_single
    
    net.train()
    return net;
#enddef


def preprocess_image(image_pyca, histeq):
    image_np = common.AsNPCopy(image_pyca)
    nan_mask = np.isnan(image_np)
    image_np[nan_mask] = 0
    image_np /= np.amax(image_np)

    # perform histogram equalization if needed
    if histeq:
        image_np[image_np != 0] = exposure.equalize_hist(image_np[image_np != 0])

    return image_np


def write_result(result, output_prefix):
    common.SaveITKImage(result['I1'], output_prefix+"I1.mhd")
    common.SaveITKField(result['phiinv'], output_prefix+"phiinv.mhd")
    I1=common.AsNPCopy(result['I1'])
    phiinv=common.AsNPCopy(result['phiinv'])
    I1_inv=common.AsNPCopy(result['I1_inv'])
    scipy.io.savemat(output_prefix+'res.mat', {'I1':I1,'I1_inv': I1_inv,'phiinv':phiinv})
#enddef


#perform deformation prediction
def predict_image(args):
    if (args.use_CPU_for_shooting):
        mType = ca.MEM_HOST
    else:
        mType = ca.MEM_DEVICE

    # load the prediction network
    predict_network_config = torch.load(args.prediction_parameter)
    prediction_net = create_net(args, predict_network_config);

#    batch_size = args.batch_size
#    patch_size = predict_network_config['patch_size']
#    print(patch_size)
#    input_batch = torch.zeros(batch_size, 2, patch_size, patch_size, 1).cuda()
#
#    # use correction network if required
#    if args.use_correction:
#        correction_network_config = torch.load(args.correction_parameter);
#        correction_net = create_net(args, correction_network_config);
#    else:
#        correction_net = None;

    # start prediction
    for i in range(0, len(args.moving_image)):

        common.Mkdir_p(os.path.dirname(args.output_prefix[i]))
        if (args.affine_align):
            # Perform affine registration to both moving and target image to the ICBM152 atlas space.
            # Registration is done using Niftireg.
            call(["reg_aladin",
                  "-noSym", "-speeeeed", "-ref", args.atlas ,
                  "-flo", args.moving_image[i],
                  "-res", args.output_prefix[i]+"moving_affine.nii",
                  "-aff", args.output_prefix[i]+'moving_affine_transform.txt'])

            call(["reg_aladin",
                  "-noSym", "-speeeeed" ,"-ref", args.atlas ,
                  "-flo", args.target_image[i],
                  "-res", args.output_prefix[i]+"target_affine.nii",
                  "-aff", args.output_prefix[i]+'target_affine_transform.txt'])

            moving_image = common.LoadITKImage(args.output_prefix[i]+"moving_affine.nii", mType)
            target_image = common.LoadITKImage(args.output_prefix[i]+"target_affine.nii", mType)
        else:
            moving_image = common.LoadITKImage(args.moving_image[i], mType)
            target_image = common.LoadITKImage(args.target_image[i], mType)

        #preprocessing of the image
#        moving_image_np = preprocess_image(moving_image, args.histeq);
#        target_image_np = preprocess_image(target_image, args.histeq);
        moving_image_np = common.AsNPCopy(moving_image);
        target_image_np = common.AsNPCopy(target_image);
        grid = moving_image.grid()
        #moving_image = ca.Image3D(grid, mType)
        #target_image = ca.Image3D(grid, mType)
#        moving_image_processed = common.ImFromNPArr(moving_image_np, mType)
#        target_image_processed = common.ImFromNPArr(target_image_np, mType)
        moving_image.setGrid(grid)
        target_image.setGrid(grid)

        # Indicating whether we are using the old parameter files for the Neuroimage experiments (use .t7 files from matlab .h5 format)
        predict_transform_space = False
        if 'matlab_t7' in predict_network_config:
            predict_transform_space = True
        # run actual prediction
#        moving_image_var=Variable(torch.from_numpy(moving_image_np).view(-1,1,192,192).cuda())
        target_image_var=Variable(torch.from_numpy(target_image_np).view(-1,1,args.image_size_nx,args.image_size_ny).cuda())        
        noiselevel = 0.1
        mask = scipy.io.loadmat(args.mask)['k']
        maski=np.fft.ifftshift(mask)
        M=pymrirecon.mriroplib(mask=maski,N=args.image_size_nx,mu=0.4,rho=0.0,lam=0.08, N_iter=20, CG_iter=20,CG_tol=1e-9,Min_iter=10)     
        Y_part = M.genksdata(moving_image_np.reshape((1,args.image_size_nx,args.image_size_ny)), noiselevel = noiselevel)  

    

        mvrc0_batch_np = M.recon_admm_batch(Y_part, np.zeros_like(moving_image_np))
        mvrc0_part_var= Variable(torch.from_numpy(mvrc0_batch_np).view(-1,1,args.image_size_nx,args.image_size_ny).cuda())
        Y_part_t = torch.from_numpy(Y_part)  
        Y_batch_var = Variable(Y_part_t.view(-1,2,args.image_size_nx,args.image_size_ny).cuda())        
#        mvrc0_part_var = Variable(torch.from_numpy(moving_image_np).view(-1,1,192,192).cuda())
        
        input_batch_variable = (mvrc0_part_var, target_image_var,Y_batch_var)
        (m0p1_var, mvp1_var, m0p2_var, mvp2_var, m0p3_var, mvp3_var, mvrc1_var, mvrc2_var, mvrc3_var) = prediction_net(input_batch_variable)#util.predict_momentum(moving_image_np, target_image_np, input_batch, batch_size, patch_size, prediction_net, predict_transform_space);
        
#        m0p1_var= torch.cat((m0p1_var,m0p1_var,m0p1_var),1)
#        m0p2_var=  m0p1_var
        
        m0 = m0p3_var.detach().view(2,args.image_size_nx,args.image_size_ny,1).cpu().numpy()#prediction_result['image_space']
        m0psli3 = np.zeros((1,args.image_size_nx,args.image_size_ny,1),dtype='float32')
        m0 = np.concatenate((m0,m0psli3),0)
        m0 = np.transpose(m0, [1, 2, 3, 0])
        m01 = m0p1_var.detach().view(2,args.image_size_nx,args.image_size_ny,1).cpu().numpy()#prediction_result['image_space']
        m01t =  m01        
        m01 = np.concatenate((m01,m0psli3),axis=0)
        m01 = np.transpose(m01, [1, 2, 3, 0])
        m02 = m0p2_var.detach().view(2,args.image_size_nx,args.image_size_ny,1).cpu().numpy()#prediction_result['image_space']       
        m02 = np.concatenate((m02,m0psli3),axis=0)
        m02 = np.transpose(m02, [1, 2, 3, 0])     
        
        mvimgrc1 = mvrc1_var.detach().view(args.image_size_nx,args.image_size_ny).cpu().numpy()  
        mvimgrc2 = mvrc2_var.detach().view(args.image_size_nx,args.image_size_ny).cpu().numpy()  
        mvimgrc3 = mvrc3_var.detach().view(args.image_size_nx,args.image_size_ny).cpu().numpy()         
        mvimgp1 = mvp2_var.detach().view(args.image_size_nx,args.image_size_ny).cpu().numpy()          
        mvimgp2 = mvp2_var.detach().view(args.image_size_nx,args.image_size_ny).cpu().numpy()  
        mvimgp3 = mvp2_var.detach().view(args.image_size_nx,args.image_size_ny).cpu().numpy()          
        #convert to registration space and perform registration
#        print(prediction_result.size(),m0.shape)
        
        m0_reg = common.FieldFromNPArr(m0, mType);
        #perform correction
#        if (args.use_correction):
#            registration_result = registration_methods.geodesic_shooting(moving_image_processed, target_image_processed, m0_reg, args.shoot_steps, mType, predict_network_config)
#            target_inv_np = common.AsNPCopy(registration_result['I1_inv'])
#
#            correct_transform_space = False
#            if 'matlab_t7' in correction_network_config:
#                correct_transform_space = True
#            correction_result = util.predict_momentum(moving_image_np, target_inv_np, input_batch, batch_size, patch_size, correction_net, correct_transform_space);
#            m0_correct = correction_result['image_space']
#            m0 += m0_correct;
#            m0_reg = common.FieldFromNPArr(m0, mType);

        registration_result = registration_methods.geodesic_shooting(moving_image, target_image, m0_reg, args.shoot_steps, mType, predict_network_config)
        I1_rec=common.AsNPCopy(registration_result['I1'])
        phiinv=common.AsNPCopy(registration_result['phiinv'])
        I1_rec_inv=common.AsNPCopy(registration_result['I1_inv'])
        
        M.__init__(mask=maski,N=args.image_size_nx,mu=0.4,rho=0.4,lam=0.08, N_iter=20, CG_iter=20,CG_tol=1e-9,Min_iter=10)
#        M=pymrirecon.mriroplib(mask=maski,N=args.image_size_nx,mu=0.4,rho=0.4,lam=0.08, N_iter=20, CG_iter=20,CG_tol=1e-9,Min_iter=10)
#        print(mvimgp3.shape)
        mv_rec_w_mvp3 = M.recon_admm_batch(Y_part, mvimgp3)
        mv_rec_w_m0p3 = M.recon_admm_batch(Y_part, I1_rec)
        scipy.io.savemat(args.output_prefix[i]+'_'+args.mask[-18:-4]+'.mat', {'m0_p3': m0,'m0_p1': m01, 'm0_p2': m02,'m0p1t': m01t,  
                         'mv_p1' : mvimgp1,'mv_p2' : mvimgp2, 'mv_p3' : mvimgp3,
                         'moving_img' : moving_image_np, 'moving_img_admm':mvrc0_batch_np,
                         'mv_rc1' : mvimgrc1,'mv_rc2' : mvimgrc2, 'mv_rc3' : mvimgrc3,
                         'I1_rec' : I1_rec,'I1_rec_inv' : I1_rec_inv, 'phiinv' : phiinv, 
                         'mv_rec_w_mvp3' : mv_rec_w_mvp3.reshape((args.image_size_nx,args.image_size_ny)), 'mv_rec_w_m0p3' : mv_rec_w_m0p3.reshape((args.image_size_nx,args.image_size_ny)),
                         'target_img':target_image_np,
                         'rho1': prediction_net.invblk_1.M.rho,'rho2': prediction_net.invblk_2.M.rho,'rho3': prediction_net.invblk_3.M.rho})



    
    
        write_result(registration_result, args.output_prefix[i]);




if __name__ == '__main__':
    check_args(args);
    predict_image(args)
