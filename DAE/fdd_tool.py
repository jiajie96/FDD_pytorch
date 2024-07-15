import datetime
from PIL import Image
from math import floor
import numpy as np
import time
from functools import partial
from random import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize
import matplotlib
import cv2
from numpy import asarray
from numpy.linalg import norm
from scipy.linalg import sqrtm
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
import torch
import cv2
from tqdm import tqdm

from Final_Github_FDD_Code.model import AutoEncoder, AutoEncoderConfig
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return asarray(images_list)


class Normalize2D(tf.keras.Model):
    """Normalize inputs at height, width channels.
    """
    def __init__(self, eps=1e-8):
        super(Normalize2D, self).__init__()
        self.eps = eps

    def call(self, x):
        """Normalize inputs.
        Args:
            x: tf.Tensor, [B, H, W, C], 2D input tensor.
        Returns:
            tf.Tensor, [B, H, W, C], normalized tensor.
        """
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return (x - mean) / tf.maximum(tf.math.sqrt(var), self.eps)



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
    

def extract_activations(loader, model, device):
    model.eval()
    model= model.to(device)
    all_latent_vectors = []
    with torch.no_grad():
        for images in tqdm(loader, desc="Extracting features"):
            images = images.to(device)
            latent_vectors = model.encode(images).cpu().numpy()
            all_latent_vectors.append(latent_vectors)

        all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    return all_latent_vectors




def calculate_fdd(images1, images2, _with_color=False, image_shape = 299):
    with open("goodluck.txt") as f:
        for line in f:
            print(line, end="")

    
    
    # Load the model    
    config = AutoEncoderConfig()
    model = AutoEncoder(config, ckpt="/FDD_pytorch/Checkpoints/checkpoint_epoch_76.pth")
    model.eval()
    print(" The pretrained DAE is uploaded")
    
    
    # Preprocess the Inputs
    dataset1 = CustomDataset_ImagenetDAE(images=images1, transform=transform_Imagenet)
    loader1 = DataLoader(dataset1, batch_size=64, shuffle=False)

    dataset2 = CustomDataset_ImagenetDAE(images=images2, transform=transform_Imagenet)
    loader2 = DataLoader(dataset2, batch_size=64, shuffle=False)

    # Extract the Activations
    act1 = extract_activations(loader1, model, device)
    act2 = extract_activations(loader2, model, device)

    
    # calculate mean and covariance statistics
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
    fdd = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fdd
