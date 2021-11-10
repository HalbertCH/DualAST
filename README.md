# DualAST: Dual Style-Learning Networks for Artistic Style Transfer
This is the official Tensorflow implementation of our paper: ["DualAST: Dual Style-Learning Networks for Artistic Style Transfer"](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_DualAST_Dual_Style-Learning_Networks_for_Artistic_Style_Transfer_CVPR_2021_paper.pdf) (**CVPR 2021**)  
  
This project provides a novel style transfer framework, termed as DualAST, to address the artistic style transfer problem from a new perspective. Unlike existing style transfer methods, which learn styles from either a single style example or a collection of artworks, DualAST learns simultaneously both the holistic artist-style (from a collection of an artist's artworks) and the specific artwork-style (from a single style image): the first style sets the tone (*i.e.*, the overall feeling) for the stylized image, while the second style determines the details of the stylized image, such as color and texture. Moreover, we introduce a Style-Control Block (SCB) to adjust the styles of generated images with a set of learnable style-control factors.  
  
![image](https://github.com/HalbertCH/DualAST/blob/main/results/1.png)  
  
## Requirements  
We recommend the following configurations:  
- python 3.7
- tensorflow 1.14.0
- CUDA 10.1
- PIL, numpy, scipy
- tqdm
  
## Model Training  
- Download the content dataset: [Places365(105GB)](http://data.csail.mit.edu/places/places365/train_large_places365standard.tar).
- Download the style dataset: [Artworks of Different Artists](https://drive.google.com/drive/folders/1WxWxIhqqtkx4CwBVem7ZSr_ay9JJCiOh?usp=sharing). Thanks for the dataset provided by [AST](https://github.com/CompVis/adaptive-style-transfer).
- Download the pre-trained [VGG-19](https://drive.google.com/drive/folders/1n7VazSzdVdAN8Bp392KYQGVshg9pTdQ4?usp=sharing) model, and record the path of VGG-19 in *vgg19.py*.
- Set your available GPU ID in Line185 of the file ‘main.py’.
- Run the following command:
```
python main.py --model_name van-gogh \
               --phase train \
               --image_size 768 \
               --ptad /disk1/chb/data/vincent-van-gogh_road-with-cypresses-1890 \
               --ptcd /disk1/chb/data/data_large
```
  
## Model Testing
- Put your trained model to *./models/* folder.
- Put some sample photographs to *./images/content/* folder.
- Put some reference images to *./images/reference/* folder.
- Set your available GPU ID in Line185 of the file ‘main.py’.
- Run the following command:
```
python main.py --model_name=van-gogh \
               --phase=inference \
               --image_size=1280 \
               --ii_dir images/content/ \
               --reference images/reference/van-gogh/1.jpg \
               --save_dir=models/van-gogh/inference
```
![image](https://github.com/HalbertCH/DualAST/blob/main/results/2.png) 
  
We provide some pre-trained models in [link](https://drive.google.com/drive/folders/1n7VazSzdVdAN8Bp392KYQGVshg9pTdQ4?usp=sharing).  
We refer the reader to [AST](https://github.com/CompVis/adaptive-style-transfer) for the computation of [Deception Rate](https://github.com/CompVis/adaptive-style-transfer/tree/master/evaluation).  
  
## Comparison Results
We compare our DualAST with [Gatys *et al.*](https://github.com/anishathalye/neural-style), [AdaIN](https://github.com/naoto0804/pytorch-AdaIN), [WCT](https://github.com/eridgd/WCT-TF), [Avatar-Net](https://github.com/LucasSheng/avatar-net), [SANet](https://github.com/GlebBrykin/SANET), [AST](https://github.com/CompVis/adaptive-style-transfer), and [Svoboda *et al.*](https://github.com/nnaisense/conditional-style-transfer).  
  
![image](https://github.com/HalbertCH/DualAST/blob/main/results/3.png)  

## Acknowledgments
The code in this repository is based on [AST](https://github.com/CompVis/adaptive-style-transfer). Thanks for both their paper and code.
