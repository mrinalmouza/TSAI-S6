
# Handwritten Digit Classifier

This is a deep neural network classiifier that uses Expand and squeeze architecture with 15K parameters.

## MNIST Dataset

MNIST dataset contains gray scale images of size 28 * 28.
The train data has 60000 images and test has 10000 images.
Below is the sample data

![alt text](https://github.com/mrinalmouza/TSAI_S6/blob/main/output.png "Sample Data")

## Architecture Diagram
Architecure diagram is below:

![alt text](https://github.com/mrinalmouza/TSAI_S6/blob/main/MNIST_20K_Mrinal.png "Architecure Diagram")

## Requirements
* matplotlib==3.7.1
* matplotlib-inline==0.1.6
* torch==2.0.1
* torchsummary==1.5.1
* torchvision==0.15.2
* tqdm==4.65.0

## Execution
To run the code, execute the S6_Assignment_Solution.ipynb file.
The model is set to execute on MPS/GPU/CPU
The training time on MPS is ~ 6 mins 

## Model Accuracy and Loss
The model reached and accuracy of 99.55% on test set with total params as 15654
The loss function used here was Crossentropy loss.

## Model specs
* Total number of trainable model parameters = 15654
* Total number of non-trainable model parameters = 0
* Model Size = 1.99MB

## Authors

- [@mrinalmouza](https://github.com/mrinalmouza)


