

# FDD_pytorch
Official Implementation of our Paper "[Enhancing Plausibility Evaluation for Generated Designs with Denoising Autoencoder](https://arxiv.org/abs/2403.05352)".

![diagram](https://github.com/jiajie96/FDD_pytorch/blob/main/Data/diagram.png)

:technologist: To calculate the FDD score between two sets of images:
```
from DAE.fdd_tool import calculate_fdd

# set_1_images and set_2_images contain respectively original data and generated data (shape of (N, H, W, C)).

fdd_score = calculate_fdd(set_1_images, set_2_images)
print('the Frechet denoised distance is:', fdd_score)
``` 

## :file_folder: Dataset
Make sure to import and save the dataset under the folder `Data/`
- The BIKED dataset is accessible via : https://decode.mit.edu/projects/biked/ 
- The ImageNet dataset is accessible via : https://www.image-net.org/download.php 
- The FFHQ dataset can be obtained from: https://github.com/NVlabs/ffhq-dataset 
- Seeing3DChairs (check again with Jiajie)


## :test_tube: Sensitivity Test
[See the main paper](https://arxiv.org/abs/2403.05352)
We provide the implementation of the levels of various disturbances, together with the distance metrics FID, FDD, TD and FD_Dino
  
## :link: Cite The Paper
If you find our work or code helpful, or your research benefits from this repo, please cite our paper:
```
@article{fan2024enhancing,
  title={Enhancing Plausibility Evaluation for Generated Designs with Denoising Autoencoder},
  author={Fan, Jiajie and Trigui, Amal and B{\"a}ck, Thomas and Wang, Hao},
  journal={arXiv preprint arXiv:2403.05352},
  year={2024}
}
``` 

  
