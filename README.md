

# FDD_pytorch
Official Implementation of our Paper "Enhancing Plausibility Evaluation for Generated Designs with Denoising Autoencoder".

![diagram](https://github.com/jiajie96/FDD_pytorch/blob/main/Data/diagram.png)

## :file_folder: Dataset
Make sure to import and save the dataset under the folder `Data/`
- The BIKED dataset is accessible via : https://decode.mit.edu/projects/biked/ 
- The ImageNet dataset is accessible via : https://www.image-net.org/download.php 
- The FFHQ dataset can be obtained from: https://github.com/NVlabs/ffhq-dataset 
- Seeing3DChairs (check again with Jiajie)

## :technologist: Denoising Autoencoder DAE
To train a new DAE model from scratch, simply run 
```
train .. 
``` 
We also provide the weights of our trained model in `Weights/`

To run inference on the validation or test set, run 
```
test .. 
``` 
## :test_tube: Sensitivity Test
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

  
