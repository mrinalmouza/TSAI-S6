# Part 1
## Model explanation:

The model shown has 2 layers(excluding the input layer). The activation used in model is sigmoid

* Forward pass:
    >> In forward pass parameters are calculated in following way:
        h1 = i1 * w1 + i2 * w2
        h2 = i1 * w3 + i2 * w4
        a_h1 = sigmoid(h1)
        a_h2 = sigmoid(h2)
        o1 = a_h1 * w5 + a_h2 * w6
        o2 = a_h1 * w7 + a_he * w8
        a_o1 = sigmoid(o1)
        a_o2 = sigmoid(o2)
        E1 = 1/2*(t1 - a_o1)^2
        E2 = 1/2*(t2 - a_o2)^2
        Etotal = E1 + E2

* Backward pass:
    >> In backward pass the derivative of Etotal is calculated w.r.t to all the weights viz -a - viz (w1,w2,... w8)
        The detailed calculation and chain rule is explained in the excel sheet.

## Model excel output
![alt text](https://github.com/mrinalmouza/TSAI-S6/Images/blob/main/Output.png "LR_0_1")
 
## Learning rate = 0.1
![alt text](https://github.com/mrinalmouza/TSAI-S6/Images/blob/main/LR_0_1.png "LR_0_1")

## Learning rate = 0.2
![alt text](https://github.com/mrinalmouza/TSAI-S6/Images/blob/main/LR_0_2.png "LR_0_2")

## Learning rate = 0.5
![alt text](https://github.com/mrinalmouza/TSAI-S6/Images/blob/main/LR_0_5.png "LR_0_5")

## Learning rate = 0.8
![alt text](https://github.com/mrinalmouza/TSAI-S6/Images/blob/main/LR_0_8.png "LR_0_8")

## Learning rate = 1.0
![alt text](https://github.com/mrinalmouza/TSAI-S6/Images/blob/main/LR_1_0.png "LR_1_0")

## Learning rate = 2.0
![alt text](https://github.com/mrinalmouza/TSAI-S6/Images/blob/main/LR_2_0.png "LR_2_0")


# PART 2

# Handwritten Digit Classifier

This is a deep neural network classiifier that uses Expand and squeeze architecture with 15K parameters.

## MNIST Dataset

MNIST dataset contains gray scale images of size 28 * 28.
The train data has 60000 images and test has 10000 images.
Below is the sample data

![alt text](https://github.com/mrinalmouza/TSAI-S6/blob/main/output.png "Sample Data")

## Architecture Diagram
Architecure diagram is below:

![alt text](https://github.com/mrinalmouza/TSAI-S6/blob/main/MNIST_20K_Mrinal.png "Architecure Diagram")

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


