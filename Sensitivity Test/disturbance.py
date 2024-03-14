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
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


def apply_salt_pepper_noise(original_data):
  
    disturbed_images = []
    for i, w in enumerate([0.03]): # to apply levels set i, w in enumerate([0.01, 0.02, 0.03])
        disturbed_images_per_level = []
        for j in range(original_data.shape[0]):
        
            img = original_data[j]
            num_pepper = np.ceil(w * img.size).astype(int)
            coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
            img_copy = img.copy()
            img_copy[coords[0], coords[1], :] = 0
            '''
            
            coords = (np.random.randint(0, img.shape[0], num_pepper),
                      np.random.randint(0, img.shape[1], num_pepper))

            img_copy = img.copy()
            img_copy[coords] = 0
            '''
            disturbed_images_per_level.append(img_copy)
            if j == 0 or j == 4 or j == 8 or j == 12:
                Image.fromarray((img_copy).astype(np.uint8)).save(f'sp_noisy_image{j}.png')
        disturbed_images.append(disturbed_images_per_level)
        
    return disturbed_images

def apply_gaussian_noise(original_data, save = True):
  
    disturbed_images = []
    noise =  np.random.normal(loc=0, scale=1, size=original_data[0].shape)
    noise = (noise*255.0).astype(numpy.uint8)
    
    for i, w in enumerate([0.2]): # to apply levels set i, w in enumerate([ 0.1, 0.2, 0.3])
        disturbed_images_per_level = []
        for j in range(original_data.shape[0]):
            img_copy = original_data[j].copy()
            noisy = (1 - w)*img_copy + w*noise
            disturbed_images_per_level.append(noisy) 
            
            if save == True:
                if j == 0 or j == 4 or j == 8 or j == 12:
                    Image.fromarray((noisy).astype(np.uint8)).save(f'gaussian_noisy_image{j}.png')
        disturbed_images.append(disturbed_images_per_level)
        
    return disturbed_images


def apply_swirl(original_data):
  
    disturbed_images = []
    for i, w in enumerate([2]): # to apply levels set i, w in enumerate([ 2, 3, 4])
        disturbed_images_per_level = []
        
        for j in range(original_data.shape[0]):
            img_copy = original_data[j].copy()
            sign = np.sign(np.random.rand(1) - 0.5)[0] # directions random
            # positioning center
            xpos = 200 // 2
            ypos = 256 // 2
            center = (xpos, ypos)
            img_s = swirl(img_copy, rotation=0, strength=sign*w, radius=120, center=center)
            disturbed_images_per_level.append(img_s*255) # or (img_copy).astype(numpy.uint8)
            if j == 0 or j == 4 or j == 8 or j == 12:
                Image.fromarray((img_s*255).astype(np.uint8)).save(f'swirled_image{j}.png')
                
        disturbed_images.append(disturbed_images_per_level)
               
    return disturbed_images

def apply_swap(original_data, save = True):
    
    left_upper_x = 0
    left_upper_y = 0
    right_lower_x = 256
    right_lower_Y = 256 
    num_cols = 8
    num_rows = 8    
    level = 1

    disturbed_images = []
    for ii, w in enumerate([1]):
        disturbed_images_per_level = []
        for jj in range(original_data.shape[0]):
            new_image = original_data[jj].copy()    
            new_image = Image.fromarray(new_image)
            focus_area = new_image.crop((left_upper_x, left_upper_y, right_lower_x, right_lower_Y))  

            width, height = focus_area.size
            num_squares= num_rows * num_cols
            square_width = width // num_cols
            square_height = height// num_rows

            squares = []
            for i in range(num_rows):  # Two rows
                for j in range(num_cols):  # Four columns
                    box = (j * square_width, i * square_height, (j + 1) * square_width, (i + 1) * square_height)
                    square = focus_area.crop(box)
                    squares.append(square)
            
            
            # swap 4 sqaures out of 16 randomly to macth the patch exchange
            num_squares_to_swap = 4
            selected_indices = random.sample(range(num_rows * num_cols), num_squares_to_swap * 2)
            for i in range(0, len(selected_indices), 2):
                squares[selected_indices[i]], squares[selected_indices[i + 1]] = squares[selected_indices[i + 1]], squares[selected_indices[i]]
            
            new_image_second = Image.new('RGB', (width, height))
   
            for i in range(num_rows):
                for j in range(num_cols):
                    box = (j * square_width, i * square_height)
                    new_image_second.paste(squares[i * num_cols + j], box)

            new_image.paste(new_image_second, (left_upper_x, left_upper_y))
            new_image = np.array(new_image)
        
            disturbed_images_per_level.append(new_image) 
            if save == True:
                if jj == 0 or jj == 4 or jj == 8 or jj == 12:
                    Image.fromarray((new_image).astype(np.uint8)).save(f'swap_L1_image{jj}.png')
        disturbed_images.append(disturbed_images_per_level)
        
    return disturbed_images




def apply_swap_GN(original_data):
    swapped =  apply_swap(original_data, save = False)
    disturbed_images = apply_gaussian_noise(np.asarray(swapped), save = False)
    
    Image.fromarray((disturbed_images[0][0][0]).astype(np.uint8)).save(f'swap_GN_image{0}.png')  
    Image.fromarray((disturbed_images[0][0][4]).astype(np.uint8)).save(f'swap_GN_image{4}.png')   
    Image.fromarray((disturbed_images[0][0][8]).astype(np.uint8)).save(f'swap_GN_image{8}.png')   
    Image.fromarray((disturbed_images[0][0][12]).astype(np.uint8)).save(f'swap_GN_image{12}.png')
    return disturbed_images[0]


def apply_mask(original_data, save = True):
   
    left_upper_x = 0
    left_upper_y = 0
    right_lower_x = 256
    right_lower_Y = 256 
    num_cols = 8
    num_rows = 8
    
    disturbed_images = []
    for ii, w in enumerate([1]): 
        disturbed_images_per_level = []
        for jj in range(original_data.shape[0]):
            new_image = original_data[jj].copy()
            new_image = Image.fromarray(new_image)
            focus_area = new_image.crop((left_upper_x, left_upper_y, right_lower_x, right_lower_Y))  

            width, height = focus_area.size
            num_squares= num_rows * num_cols
            square_width = width // num_cols
            square_height = height// num_rows

            square_coords = [(j * square_width, i * square_height, (j + 1) * square_width, (i + 1) * square_height)
                for i in range(num_rows) for j in range(num_cols)]

            # Randomly select four squares to mask
            selected_squares = random.sample(square_coords, 8)

            for box in selected_squares:
               # Create a white box with the same mode as the focus_area
               white_box = Image.new(focus_area.mode, (square_width, square_height), 'white')
               # Paste the white box onto the focus area at the box's coordinates
               focus_area.paste(white_box, box[:2])

            # Paste the updated focus area back onto the original image
            new_image.paste(focus_area, (left_upper_x, left_upper_y))            
            new_image = np.array(new_image)
            disturbed_images_per_level.append(new_image) # or (img_copy).astype(numpy.uint8)

            if save == True:
                if jj == 0 or jj == 4 or jj == 8 or jj == 12:
                    Image.fromarray((new_image).astype(np.uint8)).save(f'masked_L1_image{jj}.png')
        disturbed_images.append(disturbed_images_per_level)
        
    return disturbed_images
