from model import AutoEncoder, AutoEncoderConfig

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import cv2
from PIL import Image
import os
import glob
import numpy as np

import copy


# Parameters
batch_size = 128
num_epochs = 500
learning_rate = 0.001

# ************** HELP FUNCTIONS ######################

def scale_tensor_to_range(tensor, new_min=-1, new_max=1):
    orig_min = tensor.min()
    orig_max = tensor.max() 
    tensor_scaled = (tensor - orig_min) / (orig_max - orig_min) 
    tensor_rescaled = tensor_scaled * (new_max - new_min) + new_min  
    return tensor_rescaled

def save_latent_space(loader, model, epoch, device):
    model.eval()
    latent_space = []
    with torch.no_grad():
        for data in loader:
            img = data
            img = img.to(device)
            #img = img.view(img.size(0), -1) 
            img = img.permute(0, 3, 1, 2)

            encoded_features = model.encoder(img)
            latent_space.append(encoded_features.cpu())
   
    # Save the latent space to a file
    latent_space = torch.cat(latent_space, dim=0)
    torch.save(latent_space, f'latent_space_epoch_{epoch}.pt')
    model.train()

def add_pepper_noise(image, noise_level=0.05):
   
    bernoulli_distribution = torch.full_like(image, noise_level)
    mask = torch.bernoulli(bernoulli_distribution).bool()
   
    # Set the selected pixels to 0 (black) to simulate pepper noise
    image_noisy = image.clone()
    image_noisy[mask] = 0
   
    return image_noisy


def add_gaussian_noise(image, mean=0.0, std=0.5):
    gaussian_noise = torch.normal(mean, std, size=image.size()).to(device)
    image_noisy_tensor = image + gaussian_noise  
    return image_noisy_tensor


def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001

        loss = checkpoint['loss:']
        print(f"Checkpoint loaded. Resuming training from epoch {epoch} and with loss {loss}")
        return epoch
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0
    
# ************** END OF HELP FUNCTIONS ######################
  

    
    
# Compose the custom transformation 
transform = transforms.Compose([
   transforms.Resize((299, 299)),  
   transforms.ToTensor(),    
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

class CustomDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image



# 1- LOAD THE DATA
image_directory = 'Data/imagenet/train'
image_paths = glob.glob(os.path.join(image_directory, '*.JPEG'))
train_dataset = CustomDataset(image_paths=image_paths, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


val_image_directory = 'Data/imagenet/val'
val_image_paths = glob.glob(os.path.join(val_image_directory, '*.JPEG')) 
val_dataset = CustomDataset(image_paths=val_image_paths, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 2- CONFIGURE THE MODEL
config= AutoEncoderConfig()
config.input_dim = 299
config.input_channel = 3
model = AutoEncoder(config)
print(model)
print(config)


# 3- Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

log_file_path = 'output_log.txt'

# 4- SET LOSS AND OPTIMIZER
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
'''
optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            learning_rate,
            momentum=0.9,
            weight_decay=1e-4) 

'''
# 5- TRAINING LOOP
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1000

resume_training = False  # change it to True if you wish to resume the training from a previous saved checkpoint

# Checkpoint path
checkpoint_path = 'Checkpoints/checkpoint_epoch_76.pth'

if resume_training:
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
else:
    start_epoch = 0


train_Losses = []
val_Losses = []

for epoch in range(start_epoch, num_epochs):
    print('start of epoch:' , epoch)
    model.train()  # Set model to training mode
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        img= data
        img = img.to(device)
        images_noisy = torch.stack([add_gaussian_noise(image) for image in img])
        # Forward pass
        output = model(img)
        loss = criterion(output, img)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * img.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_Losses.append(epoch_loss)
        
    # Validation phase
    model.eval()  # Set model to evaluate mode
    val_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img_val = data
            img_val = img_val.to(device)
            images_noisy_val = torch.stack([add_gaussian_noise(image) for image in img_val])
            output_val = model(img_val)
            loss_val = criterion(output_val, img_val)
            val_loss += loss_val.item() * img_val.size(0)

    val_loss /= len(val_loader.dataset)
    val_Losses.append(val_loss)

    # Save the best model
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
          
    if epoch % 10 == 0:
        message =  f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}'
        with open(log_file_path, 'a') as log_file:
            log_file.write(message + '\n')  
            log_file.flush()  

    
    if epoch % 5 == 0: # save the checkpoint in case the run crahses !!
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch:' : epoch,
            'loss:' : epoch_loss,
        }
        torch.save(checkpoint, f'/checkpoint_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved at epoch {epoch + 1}")
    # supplementary:
    # Save latent space every n epochs
    #if (epoch + 1) % save_interval == 0:
    #   save_latent_space(train_loader, model, epoch + 1, device)
        
# Save the final model
torch.save(best_model_wts, 'Restult/DAE.pth')
np.save('Restult/train_loss.npy',train_Losses)
np.save('Restult/val_loss.npy',val_Losses)