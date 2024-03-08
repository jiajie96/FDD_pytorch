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
from torchvision import datasets, transforms
from torch.utils.data import Dataset


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2


import numpy as np
import os
import gc
import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram

from representation import get_representations




def get_train_data(data):
    image_set=[]
    for i in range(len(data)):
        image = data[i]
        image= image/127.5 -1 
        image_set.append(image)

    x_train = np.array(image_set, dtype=np.float32)
    x_train = x_train[..., np.newaxis]
    return x_train




def gaussian_kernel_matrix(x, y, sigma):
    beta = -1. / (2. * sigma ** 2)
    dist_matrix = torch.cdist(x, y)**2
    kernel_matrix = torch.exp(beta * dist_matrix)
    return kernel_matrix

def compute_mmd(x, y, sigma):
   
    x_kernel = gaussian_kernel_matrix(x, x, sigma)
    y_kernel = gaussian_kernel_matrix(y, y, sigma)
    xy_kernel = gaussian_kernel_matrix(x, y, sigma)
   
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd.item()

def calculate_mmd(images1, images2, model):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = get_train_data(images1)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)    
    act1 = extract_all_images_latent_space(test_loader, model, device)
    
    second_dataset = get_train_data(images2)
    second_loader = DataLoader(second_dataset, batch_size=64, shuffle=False)    
    act2 = extract_all_images_latent_space(second_loader, model, device)
    
    # convert numpy to tensor
    act1 = torch.tensor(act1, dtype=torch.float32)
    act2 = torch.tensor(act2, dtype=torch.float32)
    sigma = 1  # The bandwidth of the Gaussian kernel
    mmd_value = compute_mmd(act1, act2, sigma)
    return mmd_value


def calculate_FD(act1, act2):

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return score


def calculate_fid(images1, images2, model_c, _with_color=False, image_shape = 299):
    images1 = scale_images(images1, (299, 299, 3))
    images2 = scale_images(images2, (299, 299, 3))    
    images1 = tf.keras.applications.inception_v3.preprocess_input(images1)
    images2 = tf.keras.applications.inception_v3.preprocess_input(images2)
    # calculate activations
    act1 = model_c.predict(images1)
    act2 = model_c.predict(images2)
    # calculate score
    fid_c = calculate_FD(act1, act2)
    del act1, act2
    gc.collect() 
    return fid_c



def calculate_fdd(images1, images2, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = get_train_data(images1)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)    
    list_latent_1 = extract_all_images_latent_space(test_loader, model, device)
    
    second_dataset = get_train_data(images2)
    print('1_', second_dataset.shape)
    second_loader = DataLoader(second_dataset, batch_size=64, shuffle=False)    
    list_latent_2 = extract_all_images_latent_space(second_loader, model, device)
    
    
    fdd = calculate_FD(list_latent_1, list_latent_2)
    del list_latent_1, list_latent_2
    gc.collect() 
    return fdd


def extract_all_images_latent_space(loader, model, device):
    model.eval()
    all_latent_vectors = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            print('2_', images.shape)
            images = images.permute(0, 3, 1, 2)
            latent_vectors = model.encode(images).cpu().numpy()
            all_latent_vectors.append(latent_vectors)
        all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    return all_latent_vectors



transform_Imagenet = transforms.Compose([
   transforms.Resize((299, 299)),  
   transforms.ToTensor(),    
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])


class CustomDataset_ImagenetDAE(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        
        if image.ndim == 2:
            # Convert grayscale to RGB by repeating the grayscale channel thrice
            image = np.stack((image,) * 3, axis=-1)

        image = Image.fromarray(image.astype(np.uint8))
        if self.transform:
            
            image = self.transform(image)
        return image
    
    
    

def calculate_fdd_dae_Imagenet(images1, images2, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    
    dataset1 = CustomDataset_ImagenetDAE(images=images1, transform=transform_Imagenet)
    loader1 = DataLoader(dataset1, batch_size=64, shuffle=False)

    dataset2 = CustomDataset_ImagenetDAE(images=images2, transform=transform_Imagenet)
    loader2 = DataLoader(dataset2, batch_size=64, shuffle=False)
        
    list_latent_1 = extract_all_images_latent_space_ImagenetDAE(loader1, model, device)
    list_latent_2 = extract_all_images_latent_space_ImagenetDAE(loader2, model, device)
    
    
    fdd = calculate_FD(list_latent_1, list_latent_2)
    del list_latent_1, list_latent_2
    gc.collect() 
    return fdd

def extract_all_images_latent_space_ImagenetDAE(loader, model, device):
    model.eval()
    all_latent_vectors = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            #images = images.permute(0, 3, 1, 2)
            latent_vectors = model.encode(images).cpu().numpy()
            all_latent_vectors.append(latent_vectors)
        all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    return all_latent_vectors





def calculate_fd_TopoAE(images1, images2, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = get_train_data(images1)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)    
    list_latent_1 = extract_all_images_latent_space_TopoAE(test_loader, model, device)
    
    second_dataset = get_train_data(images2)
    second_loader = DataLoader(second_dataset, batch_size=64, shuffle=False)    
    list_latent_2 = extract_all_images_latent_space_TopoAE(second_loader, model, device)
    
    print(list_latent_1.shape)
    fdd = calculate_FD(list_latent_1, list_latent_2)
    del list_latent_1, list_latent_2
    gc.collect() 
    return fdd


def extract_all_images_latent_space_TopoAE(loader, model, device):
    model.eval()
    all_latent_vectors = []

    with torch.no_grad():
        for images in loader:
            images = images.permute(0,3,1,2)
            images = images.to(device)
            latent_vectors = model.encode(images).cpu().numpy()
            all_latent_vectors.append(latent_vectors)
        all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    return all_latent_vectors



def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        new_image = skimage.transform.resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)


def compute_vr_persistence_diagrams_(point_cloud):
    """
   Compute the 0-dimensional persistence diagrams for a given point cloud.

   Parameters:
   - point_cloud: A numpy array of shape (n_samples, n_features).

   Returns:
   - A numpy array representing the persistence diagram.
    """
    VR = VietorisRipsPersistence(homology_dimensions=[0], infinity_values=100, metric='euclidean', max_edge_length=100, reduced_homology =False )  # Only compute 0-dimensional homology
    diagrams = VR.fit_transform(point_cloud[None, :, :]) 
    
    return diagrams[0]  # Return the persistence diagram for the point cloud


def compute_topology_distance(diagram_real, diagram_generated, len_res):
    """
   Compute the Topology Distance (TD) between two persistence diagrams.

   Parameters:
   - diagram_real: A numpy array representing the persistence diagram for the real dataset.
   - diagram_generated: A numpy array representing the persistence diagram for the generated dataset.

   Returns:
   - The Topology Distance (TD) .
    """
    lifetimes_real = diagram_real[:, 1] - diagram_real[:, 0]
    lifetimes_generated = diagram_generated[:, 1] - diagram_generated[:, 0]

    # check that life is finite
    finite_lifetimes_real = lifetimes_real[np.isfinite(lifetimes_real)]
    finite_lifetimes_generated = lifetimes_generated[np.isfinite(lifetimes_generated)]
    
    # Sort the lifetimes in descending order
    sorted_lifetimes_real = np.sort(finite_lifetimes_real)[::-1][:len_res]
    sorted_lifetimes_generated = np.sort(finite_lifetimes_generated)[::-1][:len_res]

    
   # Compute the TD as the L2 distance between the death time vectors
    td = np.linalg.norm(sorted_lifetimes_real - sorted_lifetimes_generated)

    return td



def calculate_td(images1, images2, model_c):
    
    images1 = scale_images(images1, (299, 299, 3))
    images2 = scale_images(images2, (299, 299, 3))    
    images1 = tf.keras.applications.inception_v3.preprocess_input(images1)
    images2 = tf.keras.applications.inception_v3.preprocess_input(images2)
    # calculate activations
    act1 = model_c.predict(images1)
    act2 = model_c.predict(images2)
    diagram_real = compute_vr_persistence_diagrams_(act1)
    diagram_generated = compute_vr_persistence_diagrams_(act2)
    len_res = min(len(diagram_real), len(diagram_generated))
    td_score = compute_topology_distance(diagram_real, diagram_generated, len_res)
    del act1, act2
    gc.collect()
    return td_score




transform_dino = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])


class CustomDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if image.ndim == 2:
           # Convert grayscale to RGB by repeating the grayscale channel thrice
           image = np.stack((image,) * 3, axis=-1)

        image = Image.fromarray(image.astype(np.uint8))
        if self.transform:
            image = self.transform(image)
        return image

    

def calculate_dino(images1, images2, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    dataset1 = CustomDataset(images=images1, transform=transform_dino)
    loader1 = DataLoader(dataset1, batch_size=64, shuffle=False)

    dataset2 = CustomDataset(images=images2, transform=transform_dino)
    loader2 = DataLoader(dataset2, batch_size=64, shuffle=False)
    
    list_latent_1 = get_representations(model, loader1, device, normalized=False)
    list_latent_2 = get_representations(model, loader2, device, normalized=False)
    
    
    dino_fd = calculate_FD(list_latent_1, list_latent_2)
    del list_latent_1, list_latent_2
    gc.collect()
    return dino_fd






def calculate_TD_DAE(images1, images2, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = get_train_data(images1)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)    
    act1 = extract_all_images_latent_space(test_loader, model, device)
    
    second_dataset = get_train_data(images2)
    second_loader = DataLoader(second_dataset, batch_size=64, shuffle=False)    
    act2 = extract_all_images_latent_space(second_loader, model, device)
    
    
    diagram_real = compute_vr_persistence_diagrams_(act1)
    diagram_generated = compute_vr_persistence_diagrams_(act2)
    len_res = min(len(diagram_real), len(diagram_generated))
    td_score = compute_topology_distance(diagram_real, diagram_generated, len_res)
    del act1, act2
    gc.collect()
    return td_score



def calculate_TD_DAE_Imagenet(images1, images2, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    
    dataset1 = CustomDataset_ImagenetDAE(images=images1, transform=transform_Imagenet)
    loader1 = DataLoader(dataset1, batch_size=64, shuffle=False)

    dataset2 = CustomDataset_ImagenetDAE(images=images2, transform=transform_Imagenet)
    loader2 = DataLoader(dataset2, batch_size=64, shuffle=False)
        
    act1 = extract_all_images_latent_space_ImagenetDAE(loader1, model, device)
    act2 = extract_all_images_latent_space_ImagenetDAE(loader2, model, device)
    
    
    diagram_real = compute_vr_persistence_diagrams_(act1)
    diagram_generated = compute_vr_persistence_diagrams_(act2)
    len_res = min(len(diagram_real), len(diagram_generated))
    td_score = compute_topology_distance(diagram_real, diagram_generated, len_res)
    del act1, act2
    gc.collect()
    return td_score




def calculate_mmd_imagenet(images1, images2, model):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset1 = CustomDataset_ImagenetDAE(images=images1, transform=transform_Imagenet)
    loader1 = DataLoader(dataset1, batch_size=64, shuffle=False)

    dataset2 = CustomDataset_ImagenetDAE(images=images2, transform=transform_Imagenet)
    loader2 = DataLoader(dataset2, batch_size=64, shuffle=False)
        
    act1 = extract_all_images_latent_space_ImagenetDAE(loader1, model, device)
    act2 = extract_all_images_latent_space_ImagenetDAE(loader2, model, device)
    
    
    # convert numpy to tensor
    act1 = torch.tensor(act1, dtype=torch.float32)
    act2 = torch.tensor(act2, dtype=torch.float32)
    sigma = 1  # The bandwidth of the Gaussian kernel
    mmd_value = compute_mmd(act1, act2, sigma)
    return mmd_value
