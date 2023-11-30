# CoTAN: Cortical Surface Extraction via Topological Analysis of Neuroimaging Data

import ants
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings(action='ignore')
import pandas as pd
import cv2, os, argparse, time
import numpy as np
import nibabel as nib
from scipy.io import loadmat
from util import apply_affine, save_gifti_surface
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys
sys.path.insert(1, '/gdrive/MyDrive/SurfaceExtraction/Github/CoTAN/')
from cotan import CoTAN

if __name__ == "__main__":
    
    # --------------- Load Arguments ------------- #
    parser = argparse.ArgumentParser(description="CoTAN")
    
    parser.add_argument('--data_path', default='FetalSurfaceExtraction/data/', type=str, help="directory of the input")
    parser.add_argument('--model_path', default='FetalSurfaceExtraction/model/', type=str, help="directory of the saved models")
    parser.add_argument('--save_path', default='FetalSurfaceExtraction/results/', type=str, help="directory to save the surfaces")
    parser.add_argument('--data_name', default='dhcp', type=str, help="[dhcp, ...]")
    parser.add_argument('--device', default="cuda", type=str, help="cuda or cpu")
    parser.add_argument('--step_size', default=0.02, type=float, help="integration step size")
    parser.add_argument('--n_svf', default=4, type=int, help="number of velocity fields")
    parser.add_argument('--n_res', default=3, type=int, help="number of scales")
    parser.add_argument('-weight', action='store',dest='weight',type=str, default='FetalSurfaceExtraction/model/brain_age_weights.h5', help='name of trained weight file')
    parser.add_argument('-n_slice',action='store',dest='num_slice',default=5,type=int, help='Number of training slice from a volume')
    parser.add_argument('-gpu',action='store',dest='num_gpu',default='0', type=str, help='GPU selection')
    parser.add_argument('-batch_size', action='store',dest='bsize',default=32, type=int, help='[option] batch_size e.g. 30')
    parser.add_argument('--fix_path', default='FetalSurfaceExtraction/template/mni152_brain_clip.nii.gz',type=str, help="directory of the fixed image")
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.num_gpu
    
    fix_path = args.fix_path # directory of the aligned template 
    data_path = args.data_path  # directory of the input mri
    model_path = args.model_path  # directory of the SE saved models
    save_path = args.save_path  # directory to save the surface
    data_name = args.data_name  # dhcp
    device = torch.device(args.device)
    step_size = args.step_size
    M = args.n_svf
    R = args.n_res

    # ------- Gestational Age Prediction ------- #
    
    # Loss function for age prediction
    def huber_loss(y_true, y_pred, delta=1.0 ):
        error = y_pred - y_true
        abs_error = K.abs(error)
        quadratic = K.minimum(abs_error, delta)
        linear = abs_error - quadratic
        return 0.5 * K.square(quadratic) + delta * linear

    # ------- Get the mode of the age prediction ------- #
    def _get_mode(predic_age, bin_range=0.2):
        bin_list=np.arange(np.floor(np.min(predic_age))-5,np.ceil(np.max(predic_age))+5,bin_range)
        bin_count=np.zeros([len(bin_list),])
        for i in range(0,len(predic_age)):
            bin_count[np.bitwise_and((predic_age[i]-bin_list)<=(bin_range/2),(predic_age[i]-bin_list)>=(-bin_range/2))] = bin_count[np.bitwise_and((predic_age[i]-bin_list)<=(bin_range/2),(predic_age[i]-bin_list)>=(-bin_range/2))]+1
        if np.sum(bin_count==max(bin_count))>1:
            pred_sub_argmax = 3*np.median(predic_age) - 2*np.mean(predic_age)
        else:
            j=np.where(bin_count==np.max(bin_count))[0][0]
            f=bin_count[j-1:j+2]
            L = bin_list[j]-bin_range/2
            pred_sub_argmax = L + bin_range*((f[1]-f[0])/((2*f[1]) - f[0] - f[2]))
    
        return pred_sub_argmax
    
    # ------- Crop and pad the image to the target shape ------- #
    def crop_pad_ND(img, target_shape):
        import operator, numpy as np 
        if (img.shape > np.array(target_shape)).any():
            target_shape2 = np.min([target_shape, img.shape],axis=0)
            start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2)) 
            end = tuple(map(operator.add, start, target_shape2)) 
            slices = tuple(map(slice, start, end))
            img = img[tuple(slices)]
        offset = tuple(map(lambda a, da: a//2-da//2, target_shape, img.shape))
        slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
        result = np.zeros(target_shape)
        result[tuple(slices)] = img
        return result
    
    # ------- Make the input image list to the network ------- #
    def make_dic(img, num_slice, slice_mode=0, desc=''):
        max_size = [290, 290, 1]
        target_size = (224, 224)
        if slice_mode:
            dic = np.zeros([1, max_size[1], max_size[0], num_slice], dtype=np.float16)
        else:
            dic = np.zeros([num_slice, 224, 224, 1], dtype=np.float16)
        try:
            img_data = np.squeeze(nib.load(img).get_fdata())
            assert not np.isnan(img_data).any()
            img_data = ((img_data - float(img_data.min())) / float(img_data.max() + 1E-7)) * float(255)
            img_data = crop_pad_ND(img_data, np.max(np.vstack((max_size, img_data.shape)), axis=0)) 
            img_data = cv2.resize(img_data, target_size, interpolation=cv2.INTER_LINEAR)
            img_data = img_data / 255.0
            if slice_mode: 
                dic[0, :, :, :] = np.swapaxes(
                    img_data[:, :, int(img_data.shape[-1] / 2 - 1 - np.int(num_slice / 2)):int(
                        img_data.shape[-1] / 2 + np.int(num_slice / 2))], 0, 1)
            else:
                dic[:, :, :, 0] = np.swapaxes(
                    img_data[:, :, int(img_data.shape[-1] / 2 - 1 - np.int(num_slice / 2)):int(
                        img_data.shape[-1] / 2 + np.int(num_slice / 2))], 0, 2)
        except Exception as e:
            print(f"Error loading file {img}: {e}") 
        return dic
    
    # ------- Data augmentation for age prediction ------- #
    datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=360,
        width_shift_range=0.15,
        height_shift_range=0.1,
        vertical_flip=True,
        horizontal_flip=True)
    
    # ------- TTA prediction ------- #
    def tta_prediction(datagen, model, dic, n_example):
        preds=np.zeros([len(dic),])
        for i in range(len(dic)):
            image = np.expand_dims(dic[i],0)
            pred = model.predict_generator(datagen.flow(image, batch_size=n_example),workers=4,steps=n_example, verbose=0)
            preds[i]=np.mean(pred)
        return preds
    
    # ------- Build the age prediction network ------- #
    def age_predic_network(img_shape):
        ''' Use ResNet50V2 as the backbone and add a dropout layer and a dense layer'''
        model = ResNet50V2(input_shape=img_shape,include_top=False, weights=None, pooling='avg')
        o = Dropout(0.3)(model.layers[-1].output)
        o = Dense(1,activation='linear')(o)
        model = Model(model.layers[0].output, o)
        model.compile(optimizer=Adam(learning_rate=0.001,decay=0.001), loss=huber_loss, metrics=['mae'])
        return model
    
    # ------ Inference Gestational Age ------ 
    model = age_predic_network([224,224,1])
    test_dic = make_dic(args.data_path, args.num_slice)
    model = age_predic_network([224,224,1])
    model.load_weights(args.weight)
    p_age2 = tta_prediction(datagen,model,test_dic,20)
    
    age = _get_mode(p_age2)
    print(age)
    del model
    
    # ------ Affine T2  ------- #
    def affine_matrix(ants_trans):
        """Convert ants transform to a 4x4 affine matrix"""
        transform = np.zeros([4,4])
        m_matrix = loadmat(
            ants_trans['fwdtransforms'][0])['AffineTransform_float_3_3'][:9].reshape(3,3)# .T
        m_center = loadmat(
            ants_trans['fwdtransforms'][0])['fixed'][:,0]
        m_translate = loadmat(
            ants_trans['fwdtransforms'][0])['AffineTransform_float_3_3'][9:][:,0]
        m_offset = m_translate + m_center - m_matrix @ m_center

        # ITK affine to affine matrix
        transform[:3,:3] = m_matrix
        transform[:3,-1] = -m_offset
        transform[3,:] = np.array([0,0,0,1])
    
        # LIP space to RAS
        transform[2,-1] = -transform[2,-1]
        transform[2,1] = -transform[2,1]
        transform[1,2] = -transform[1,2]
        transform[2,0] = -transform[2,0]
        transform[0,2] = -transform[0,2]
        return transform
    
    # ------- Load images ------- #
    fix_img = ants.image_read(fix_path)
    affine_fix = nib.load(fix_path).affine
    move_img = ants.image_read(data_path)

    # ------- Affine registration -------#
    ants_trans = ants.registration(
        fixed=fix_img,
        moving=move_img,
        type_of_transform='AffineFast',
        aff_metric='GC')
    
    # ------- Warp the image ------- #
    warp_img = ants.apply_transforms(
        fixed=fix_img,
        moving=move_img,
        transformlist=ants_trans['fwdtransforms'],
        interpolator='linear')

    # ------- Compute new affine matrix ------- #
    affine_mat = affine_matrix(ants_trans)
    affine_warp = affine_mat @ affine_fix

    # ------- Save File ------- # 
    warp_img = nib.Nifti1Image(
        warp_img.numpy().astype(np.float32), affine_warp)
    warp_img.header['xyzt_units']=2
    nib.save(warp_img, save_path+'affined.nii.gz')
    affined_path = save_path+'affined.nii.gz'
    print(affined_path)
    
    # ------- Surface Extraction model ------- #
    print('Load template ...')
    vol_mni = nib.load('FetalSurfaceExtraction/template/mni152_brain_clip.nii.gz')
    affine_in = vol_mni.affine
    surf_left_in = nib.load('FetalSurfaceExtraction/template/init_surf_left.surf.gii')
    surf_right_in = nib.load('FetalSurfaceExtraction/template/init_surf_right.surf.gii')
    
    # ------ Load Left Input Surfaces ------ #
    v_left_in = surf_left_in.agg_data('pointset')
    f_left_in = surf_left_in.agg_data('triangle')
    v_left_in = apply_affine(v_left_in, np.linalg.inv(affine_in))
    v_left_in[:,0] = v_left_in[:,0] - 64
    v_left_in = (v_left_in - [56, 112, 80]) / 112
    f_left_in = f_left_in[:,[2,1,0]]
    v_left_in = torch.Tensor(v_left_in[None]).to(device)
    f_left_in = torch.LongTensor(f_left_in[None]).to(device)

    # ------- Load Right Input Surfaces ------- #
    v_right_in = surf_right_in.agg_data('pointset')
    f_right_in = surf_right_in.agg_data('triangle')
    v_right_in = apply_affine(v_right_in, np.linalg.inv(affine_in))
    f_right_in = f_right_in[:,[2,1,0]]
    v_right_in = (v_right_in - [56, 112, 80]) / 112
    v_right_in = torch.Tensor(v_right_in[None]).to(device)
    f_right_in = torch.LongTensor(f_right_in[None]).to(device)
     
    # ------- Input integration time sequence ------- #
    T = torch.arange(1./step_size).to(device).unsqueeze(1) * step_size

    # ------- Load Input Volume ------- # 
    vol = nib.load(affined_path)
    affine_t2 = vol.affine 
    vol_arr = (vol.get_fdata() / 40.).astype(np.float32)  # normalize intensity
    vol_in = torch.Tensor(vol_arr[None,None]).to(device)
    vol_left_in = vol_in[:,:,64:]
    vol_right_in = vol_in[:,:,:112]

    # ------- Normalize age ------- #
    age = (age-20) / 30.  # normalize age
    age = torch.Tensor(np.array(age)[None]).to(device) # Convert to tensor
    age = age.repeat(int(1./step_size), 1).to(device)
    
    
    # ------- Initialize Surface Extraction Model ------- # 
    print('Initalize model ...')
    model_left_white = CoTAN(
        layers=[16,32,64,128,128], M=M, R=R).to(device)
    model_left_pial = CoTAN(
        layers=[16,32,32,32,32], M=M, R=R).to(device)
    model_right_white = CoTAN(
        layers=[16,32,64,128,128], M=M, R=R).to(device)
    model_right_pial = CoTAN(
        layers=[16,32,32,32,32], M=M, R=R).to(device)
    
    
    # ------ Upload Surface Extraction model ------ #
    model_left_white.load_state_dict(torch.load(
        model_path+'model_'+data_name+'_left_white.pt', map_location=device))
    model_left_pial.load_state_dict(torch.load(
        model_path+'model_'+data_name+'_left_pial.pt', map_location=device))
    model_right_white.load_state_dict(torch.load(
        model_path+'model_'+data_name+'_right_white.pt', map_location=device))
    model_right_pial.load_state_dict(torch.load(
        model_path+'model_'+data_name+'_right_pial.pt', map_location=device))
        

    # ------ Inference ------ # 
    print('Start surface reconstruction ...')
    t_start = time.time()
    with torch.no_grad():
        v_left_white = model_left_white(
            v_left_in, T, age, vol_left_in)
        v_left_pial = model_left_pial(
            v_left_white, T, age, vol_left_in)
        v_right_white = model_right_white(
            v_right_in, T, age, vol_right_in)
        v_right_pial = model_right_pial(
            v_right_white, T, age, vol_right_in)
    t_end = time.time()
    print('Finished. Runtime:{}'.format(np.round(t_end-t_start,4)))
    print('Save surface meshes ...', end=' ')
    
    # Tensor to numpy
    v_left_white = v_left_white[0].cpu().numpy()
    v_left_pial = v_left_pial[0].cpu().numpy()
    f_left_in = f_left_in[0].cpu().numpy()
    v_right_white = v_right_white[0].cpu().numpy()
    v_right_pial = v_right_pial[0].cpu().numpy()
    f_right_in = f_right_in[0].cpu().numpy()
    
    # Map surfaces to their original spaces
    v_left_white = v_left_white * 112 + [56, 112, 80]
    v_left_white[:,0] = v_left_white[:,0] + 64
    v_left_white = apply_affine(v_left_white, affine_t2)
    v_left_pial = v_left_pial * 112 + [56, 112, 80]
    v_left_pial[:,0] = v_left_pial[:,0] + 64
    v_left_pial = apply_affine(v_left_pial, affine_t2)
   
    f_left_in = f_left_in[:,[2,1,0]]

    v_right_white = v_right_white * 112 + [56, 112, 80]
    v_right_white = apply_affine(v_right_white, affine_t2)
    v_right_pial = v_right_pial * 112 + [56, 112, 80]
    v_right_pial = apply_affine(v_right_pial, affine_t2)
    f_right_in = f_right_in[:,[2,1,0]]
    
     # ------ Save surfaces ------ 
    save_gifti_surface(v_left_white, f_left_in,
                       save_path+'surf_left_white.surf.gii',
                       surf_hemi='CortexLeft', surf_type='GrayWhite')
    save_gifti_surface(v_left_pial, f_left_in,
                       save_path+'surf_left_pial.surf.gii',
                       surf_hemi='CortexLeft', surf_type='Pial')
    save_gifti_surface(v_right_white, f_right_in,
                       save_path+'surf_right_white.surf.gii',
                       surf_hemi='CortexRight', surf_type='GrayWhite')
    save_gifti_surface(v_right_pial, f_right_in,
                       save_path+'surf_right_pial.surf.gii',
                       surf_hemi='CortexRight', surf_type='Pial')
    print('Done.')