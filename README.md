# Deep joint indirect registration and reconstruction

This repository contains the codes for the paper: Rethinking medical image reconstruction via shape prior, going deeper and faster: Deep joint indirect registration and reconstruction.

# Prepare Data

The quicksilver (https://github.com/rkwitt/quicksilver) was used to generate the momentum.

# Training 


## The code train_indreg_recon_mri.py is used for training the model for joint indirect registration and MRI reconstruction  
### Example
python ./train_indreg_recon_mri.py --moving-image-dataset cdmri_dataset/dtstp1_moving_1.pth.tar cdmri_dataset/dtstp1_moving_2.pth.tar  cdmri_dataset/dtstp1_moving_3.pth.tar cdmri_dataset/dtstp1_moving_4.pth.tar cdmri_dataset/dtstp1_moving_5.pth.tar  --target-image-dataset cdmri_dataset/dtstp1_target_1.pth.tar cdmri_dataset/dtstp1_target_2.pth.tar cdmri_dataset/dtstp1_target_3.pth.tar cdmri_dataset/dtstp1_target_4.pth.tar cdmri_dataset/dtstp1_target_5.pth.tar  --deformation-parameter cdmri_dataset/dtstp1_m0_1.pth.tar cdmri_dataset/dtstp1_m0_2.pth.tar cdmri_dataset/dtstp1_m0_3.pth.tar cdmri_dataset/dtstp1_m0_4.pth.tar cdmri_dataset/dtstp1_m0_5.pth.tar --deformation-setting-file ./LDDMM_spec.yaml  --batch-size 8 --output-name prediction_mri_005_1s.pth.tar --epochs 3000   

## The code train_recon_indreg_ct.py is used for training the model for joint indirect registration and CT reconstruction  


# Inference


## The code predict_recon_indreg_mri.py is used for predict the indirect registration and reconstruct MRI images  
### Example 
python ./predict_indreg_recon_mri.py --moving-image cdmri_dataset/mv_1.nii  cdmri_dataset/mv_2.nii  cdmri_dataset/mv_3.nii   cdmri_dataset/mv_4.nii  cdmri_dataset/mv_5.nii  cdmri_dataset/mv_6.nii  cdmri_dataset/mv_7.nii  cdmri_dataset/mv_8.nii cdmri_dataset/mv_9.nii  cdmri_dataset/mv_10.nii  cdmri_dataset/mv_11.nii   cdmri_dataset/mv_12.nii  cdmri_dataset/mv_13.nii  cdmri_dataset/mv_14.nii  cdmri_dataset/mv_15.nii  cdmri_dataset/mv_16.nii --target-image  cdmri_dataset/tg_1.nii  cdmri_dataset/tg_2.nii  cdmri_dataset/tg_3.nii  cdmri_dataset/tg_4.nii cdmri_dataset/tg_5.nii  cdmri_dataset/tg_6.nii cdmri_dataset/tg_7.nii cdmri_dataset/tg_8.nii cdmri_dataset/tg_9.nii  cdmri_dataset/tg_10.nii  cdmri_dataset/tg_11.nii  cdmri_dataset/tg_12.nii cdmri_dataset/tg_13.nii  cdmri_dataset/tg_14.nii cdmri_dataset/tg_15.nii cdmri_dataset/tg_16.nii --output-prefix  res/res_d_1   res/res_d_2  res/res_d_3  res/res_d_4 res/res_d_5 res/res_d_6 res/res_d_7 res/res_d_8 res/res_d_9   res/res_d_10  res/res_d_11  res/res_d_12 res/res_d_13 res/res_d_14 res/res_d_15 res/res_d_16  --prediction-parameter  /home/jlliu/indirect_reg_recon_GPUTF_CT_MRI/prediction_mri_005_1s.pth.tar 

## The code predict_recon_indreg_ct.py is used for predict the indirect registration and reconstruct CT images   


# Citation

@article{liu2021rethinking,
  title={Rethinking medical image reconstruction via shape prior, going deeper and faster: Deep joint indirect registration and reconstruction},
  author={Liu, Jiulong and Aviles-Rivero, Angelica I and Ji, Hui and Sch{\"o}nlieb, Carola-Bibiane},
  journal={Medical Image Analysis},
  volume={68},
  pages={101930},
  year={2021},
  publisher={Elsevier}
}
