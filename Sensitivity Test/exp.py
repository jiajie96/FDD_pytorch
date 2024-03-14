'''
a python script to divide the totality of 3000 bike images , split them into 10 groups of 300, for each group then apply the same distrubance function predefined liked swirl and compare it to its original state using a distance metric, at the ened we ll get for each distrubance function a set of 10 score values , from which we track the minimun, the maxinum and the mean.
Similarly , applied to CHAIR and FFHQ dataset
'''

from utils import calculate_FD,  calculate_fid, calculate_td, calculate_fdd, calculate_fd_TopoAE, extract_all_images_latent_space, get_train_data,  scale_images, calculate_dino, calculate_TD_DAE, calculate_fdd_dae_Imagenet, calculate_mmd, calculate_TD_DAE_Imagenet, calculate_mmd_imagenet
from disturbance import apply_salt_pepper_noise, apply_gaussian_noise, apply_swirl, apply_swap, apply_swap_GN, apply_mask

import torch
import numpy as np
import cv2
import os
import glob
from PIL import Image
from scipy.linalg import sqrtm
import numpy
from skimage.transform import swirl
import random
import skimage
import tensorflow as tf
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


from DAE.model import AutoEncoder, AutoEncoderConfig
from load_encoder import load_encoder

def create_disturbed_groups(image_paths, group_size):
    
    all_original_images= []
    all_disturbed_images = []
    
    count = 1
    for i in range(0, len(image_paths), group_size):
        group_i = image_paths[i:i + group_size]
        all_disturbed_images_one_group = []

        # extract the images
        original_images= []
        for image_path in group_i:
            
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            #original_image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_images.append(original_image)

        original_images = np.asarray(original_images) 
        np.save(f'/original_group_{count}',original_images)
        all_original_images.append(original_images)
        
        
        # apply SP Noise
        SP_disturbed_images = np.asarray(apply_salt_pepper_noise(original_images))
        np.save(f'saltPepper_group_{count}',SP_disturbed_images)
        all_disturbed_images_one_group.append(SP_disturbed_images)
        
        # apply Gaussian Noise
        GN_disturbed_images = np.asarray(apply_gaussian_noise(original_images))
        np.save(f'gaussian_noise_group_{count}',GN_disturbed_images)
        all_disturbed_images_one_group.append(GN_disturbed_images)
           
        # apply swirl 
        Swirl_disturbed_images = np.asarray(apply_swirl(original_images))
        np.save(f'swirl_group_{count}',Swirl_disturbed_images)
        all_disturbed_images_one_group.append(Swirl_disturbed_images)
        
        # apply swap
        Swap_disturbed_images = np.asarray(apply_swap(original_images))
        np.save(f'swap_group_{count}',Swap_disturbed_images)
        all_disturbed_images_one_group.append(Swap_disturbed_images)
        
        
        # apply mask
        masked_disturbed_images = np.asarray(apply_mask(original_images))
        print(masked_disturbed_images.shape)
        np.save(f'mask_group_{count}',masked_disturbed_images)
        all_disturbed_images_one_group.append(masked_disturbed_images)
           
        
        #apply Swap + GN       
        Swap_GN_disturbed_images = np.asarray(apply_swap_GN(original_images))
        np.save(f'swap_GN_group_{count}',Swap_GN_disturbed_images)
        all_disturbed_images_one_group.append(Swap_GN_disturbed_images)
        
        all_disturbed_images.append(all_disturbed_images_one_group)
        count +=1
    return all_original_images, all_disturbed_images




def main():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    tf.debugging.set_log_device_placement(False)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Invalid device or cannot modify virtual devices once initialized
            print(e)
            
    image_directory = 'Data/test_biked'
    image_paths = glob.glob(os.path.join(image_directory, '*.png'))
    print(len(image_paths))
    np.random.shuffle(image_paths)

    group_size =100
    nb_groups = 10
    scores = []

    # 1- disturbe the images inside each group
    all_original_images, all_disturbed_images =  create_disturbed_groups(image_paths, group_size)
    
    # 2- Calculate scores
    
    # a- FID score
    model_c =tf.keras.applications.inception_v3.InceptionV3(weights= '/root/FD_score/inception_v3.h5',include_top=False, pooling='avg',input_shape=(299, 299, 3))
    fid_scores = [] # for all groups
    for i in range(nb_groups) : 
        original_group = all_original_images[i]
        all_disturbtions_in_group = all_disturbed_images[i]
        all_fids_4_disturbance =[]
        for f in range(6): # f :0 ,1,2,3, 4 each for one disturbtion function : SP, GN, Swirl, Swap
            one_disturbed_group = all_disturbtions_in_group[f]  # (1, group size, 256, 256) ;  1 == level of noise
            one_disturbed_group_per_level = one_disturbed_group[0] 
            score_fid = calculate_fid(original_group, one_disturbed_group_per_level, model_c)
            all_fids_4_disturbance.append(score_fid)
         
        fid_scores.append(all_fids_4_disturbance)
        
    np.save(f'/Sensitivity_Test/Experiment/scores/fid.npy', fid_scores)
     
    # b - FDD score
    checkpoint_path = 'Checkpoints/checkpoint_epoch_76.pth'
    config= ImagenetAutoEncoderConfig()
    config.input_dim = 299
    config.input_channel = 3
    model = ImagenetAutoEncoder(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    fdd_scores = [] # for all groups
    for i in range(nb_groups) : 
        original_group = all_original_images[i]
        all_disturbtions_in_group = all_disturbed_images[i]
        all_fdds_4_disturbance =[]
        for f in range(6): # f :0 ,1,2,3 each for one disturbtion function
            one_disturbed_group = all_disturbtions_in_group[f]  # (3,group size, 256, 256) ; 1 == level of noise            
            one_disturbed_group_per_level = one_disturbed_group[0] 
            score_fdd = calculate_fdd_dae_Imagenet(original_group, one_disturbed_group_per_level, model)
            all_fdds_4_disturbance.append(score_fdd)
 
        fdd_scores.append(all_fdds_4_disturbance)
          
   
    np.save(f'Sensitivity_Test/Experiment/scores/FDD.npy',fdd_scores)

    # c - TD score
    model_c =tf.keras.applications.inception_v3.InceptionV3(weights= '/root/FD_score/inception_v3.h5',include_top=False, pooling='avg',input_shape=(299, 299, 3))
    
    td_scores = [] # for all groups
    for i in range(nb_groups) : 
        original_group = all_original_images[i]
        all_disturbtions_in_group = all_disturbed_images[i]
        all_tds_4_disturbance =[]
        for f in range(6): # f :0 ,1,2,3 each for one disturbtion function : SP, GN, Swirl, Swap
            one_disturbed_group = all_disturbtions_in_group[f]  # (1, group size, 256, 256) ;  1 == level of noise
            one_disturbed_group_per_level = one_disturbed_group[0] 
            score_td = calculate_td(original_group, one_disturbed_group_per_level, model_c)
            print(score_td)
            all_tds_4_disturbance.append(score_td)
         
        td_scores.append(all_tds_4_disturbance)
        
    np.save(f'Sensitivity_Test/Experiment/scores/TD.npy,td_scores)
    
    
    # d- Dino score
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_dino = load_encoder('dinov2', device, ckpt=None, arch=None, clean_resize=True)
    
    dino_scores = [] # for all groups
    for i in range(nb_groups) : 
        original_group = all_original_images[i]
        all_disturbtions_in_group = all_disturbed_images[i]
        all_dino_Scores_4_disturbance =[]
        for f in range(6): # f :0 ,1,2,3 each for one disturbtion function : SP, GN, Swirl, Swap
            one_disturbed_group = all_disturbtions_in_group[f]  # (1, group size, 256, 256) ;  1 == level of noise
            one_disturbed_group_per_level = one_disturbed_group[0] 
            score_dino = calculate_dino(original_group, one_disturbed_group_per_level, model_dino)
            print(score_dino)
            all_dino_Scores_4_disturbance.append(score_dino)
         
        dino_scores.append(all_dino_Scores_4_disturbance)
        
    np.save(f'Sensitivity_Test/Experiment/scores/DinoFD.npy',dino_scores)
    
    '''
    # TDD on Imagenet
    checkpoint_path = 'Checkpoints/checkpoint_epoch_76.pth'

    config= ImagenetAutoEncoderConfig()
    config.input_dim = 299
    config.input_channel = 3
    model = ImagenetAutoEncoder(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    scores = [] # for all groups
    for i in range(nb_groups) : 
        original_group = all_original_images[i]
        all_disturbtions_in_group = all_disturbed_images[i]
        all_scores_4_disturbance =[]
        for f in range(6): # f :0 ,1,2,3 each for one disturbtion function
            one_disturbed_group = all_disturbtions_in_group[f]  # (3,group size, 256, 256) ; 1 == level of noise            
            one_disturbed_group_per_level = one_disturbed_group[0] 
            scores = calculate_TD_DAE_Imagenet(original_group, one_disturbed_group_per_level, model)
            all_scores_4_disturbance.append(scores)
 
        scores.append(all_scores_4_disturbance)
          
   
    np.save(f'Sensitivity_Test/Experiment/scores/TD_DAE_Imagenet_scores.npy',scores)
    
    # MMD on DAE Imagenet 
    checkpoint_path = ''

    config= ImagenetAutoEncoderConfig()
    config.input_dim = 299
    config.input_channel = 3
    model = ImagenetAutoEncoder(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    scores = [] # for all groups
    for i in range(nb_groups) : 
        original_group = all_original_images[i]
        all_disturbtions_in_group = all_disturbed_images[i]
        all_4_disturbance =[]
        for f in range(6): # f :0 ,1,2,3 each for one disturbtion function
            one_disturbed_group = all_disturbtions_in_group[f]  # (3,group size, 256, 256) ; 1 == level of noise            
            one_disturbed_group_per_level = one_disturbed_group[0] 
            score = calculate_mmd_imagenet(original_group, one_disturbed_group_per_level, model)
            all_4_disturbance.append(score)
 
        scores.append(all_4_disturbance)
   
    np.save(f'Sensitivity_Test/Experiment/scores/mdd_scores_DAE_Imagenet.npy',scores)
    '''
    
    
    fid_scores= fid_scores
    td_scores= td_scores
    fd_scores = td_scores
    dino_scores = dino_scores
    score_dict = {
        'FID': fid_scores,
        'FDD': fdd_scores,
        'TD':  td_scores,
        'DinoFD': dino_scores,
        } 
    

    return score_dict
    
    
    

if __name__ == "__main__":
    main()

