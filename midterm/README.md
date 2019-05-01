# CPSC8810Deep-Learning-Project
Midterm report deadline : March 15th  
Group members : chong meng, ziheng he

## run our model

Download VGG.py, test.py and the trained model (from Google Drive), get the predicted result by  

python test.py img_filename

## network structure

The prototype of our network structure is VGG19 with batch normalization.  
We code the VGG19 from scratch and compare it with model from pytorch library.

## training strategy

### data augmentation

For each image in the training dataset, we create 5 transformed versions of it : 1 flip and 4 different rotation.   

The ratio between training images and test images is 4:1.

### Training algorithm

We use mini-batch gradient descent method with a batch size of 32.

## code usage 
image processing : PIL lib  
model : vgg from pytorch with some modifications

## reference 

1. pytorch documentation
2. stackoverflow
3. research papers about batchnorm,learning rate,VGG
