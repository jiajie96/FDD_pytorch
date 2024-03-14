import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from DAE.model import AutoEncoder, AutoEncoderConfig
from torch.utils.data import Dataset
import glob
from PIL import Image
import os
import glob
import numpy as np
import copy
from run import scale_tensor_to_range

# Parameters
input_size = 299
hidden_size = 128
batch_size = 64

# Data loading
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




test_image_directory = 'Data/'
test_image_paths = glob.glob(os.path.join(test_image_directory, '*.png'))
test_dataset = CustomDataset(image_paths=test_image_paths, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Loading
checkpoint_path = 'Checkpoints/checkpoint_epoch_76.pth'

config= AutoEncoderConfig()
config.input_dim = 299
config.input_channel = 3
model = AutoEncoder(config)
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss function
criterion = torch.nn.MSELoss()



# Test the autoencoder on unseen data
def test_autoencoder(loader, model, device, criterion, visualize=True, num_visualizations=2):
    total_loss = 0
    with torch.no_grad():
        for i, img in enumerate(loader):
            img = img.to(device)
            output = model(img)
            loss = criterion(output, img)
            total_loss += loss.item()
           
            if visualize and i == 0:  # Visualize the first batch
                img = img.cpu().view(img.size(0),3, 299, 299)                
                output = output.cpu().view(output.size(0),3, 299, 299)
                plt.figure(figsize=(10, 4))
                for j in range(num_visualizations):
                   
                    # Original Images
                    plt.subplot(2, num_visualizations, j+1)
                    img[j]= scale_tensor_to_range(img[j])
                    plt.imshow(img[j][0], cmap='gray')
                    plt.axis('off')
                    if j == 0:
                        plt.title('Original Images')
                   
                   # Reconstructed Images
                    plt.subplot(2, num_visualizations, num_visualizations+j+1)
                    rescaled= scale_tensor_to_range(output[j][0])     
                    plt.imshow(rescaled, cmap='gray')
                    plt.axis('off')
                    if j == 0:
                        plt.title('Reconstructed Images')
               
                plt.show()
                break  

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss




# Our autoencoder has an 'encode' method that returns the latent vectors
def extract_all_images_latent_space(loader, model, device):
    model.eval()
    all_latent_vectors = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            latent_vectors = model.encode(images).cpu().numpy()
            all_latent_vectors.append(latent_vectors)
        all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    return all_latent_vectors

def visualize_test_images_latent_space(loader, model, device, num_components=2):
    model.eval()
    all_latent_vectors = extract_all_images_latent_space(loader, model, device)
   # Use PCA to reduce the dimensionality of the latent vectors
    pca = PCA(n_components=num_components)
    latent_2d = pca.fit_transform(all_latent_vectors)

   # Plot the 2D latent space
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], cmap='viridis', s=2)  
    plt.colorbar()  
    plt.title('2D PCA visualization of the latent space')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.show()

    
    


# Test the autoencoder
test_loss = test_autoencoder(test_loader, model, device, criterion, visualize=True, num_visualizations=5)
print(f'Test Loss: {test_loss:.6f}')
visualize_test_images_latent_space(test_loader, model, device, num_components=2)

latent_features = extract_all_images_latent_space(test_loader, model, device)
np.save('Result/latent.npy',latent_features )

