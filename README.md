# About this project:
While building our model, we tested numerous other pretrained models, such as Resnet,
Densenet, Imagenet, and ViT. Our initial findings indicated that Efficient Net was the most
accurate for our model. Its efficiency and accuracy was unparalleled compared to other models
that we tested. But after observing people’s success with ViT, we decided to incorporate ViT into
our model as well. As a result, we designed our model as an ensemble of ViT and Efficient Net.
Specifically, we used ViT-Huge in conjunction with EfficientNet_v2_L.
During our testing, we discovered that freezing all layers except the last few layers was an
effective way of teaching our model to learn different patterns from the limited training dataset
we were provided. We applied this strategy to each pretrained model. To reduce overfitting, we
added a dropout layer of 0.25. Despite this, overfitting was still a persistent issue.
To remedy this, we found a new tool called SAM. SAM, also known as Sharpness Aware
Minimization, helped our model with generalization by minimizing the loss value and the
sharpness of the loss. It did this by seeking parameters that lie in neighborhoods with low loss,
which factored in the shape, making the model more robust to noise.
We used stochastic gradient descent enhanced with SAM to optimize the model’s parameters. It
updates the weights using a fixed learning rate to minimize the loss function. A learning rate of
0.004 was chosen to ensure the model learns at a steady pace, reducing the risk of overshooting
the minima. Additionally, we factored in a momentum of 0.9 to accelerate convergence and reach
the global minimum efficiently. To classify the odds of an image being of a certain class, we used
Cross-Entropy Loss from the pytorch library.
For image processing, we converted the size of all training images into 480x480 to preserve
details in the image as well as to standardize input size across all data. We normalized the images
and used data augmentations, such as horizontal flips, color jittering, and vertical flips to expand
our dataset, as we were only provided with 1000 images to train with. We partitioned the training
folder into an 80 20 training and validation set using torch’s built in ImageFolder. We trained for
many epochs with a batch size of 16.
To evaluate the performance of our model, we used accuracy as well as the training and
validation loss to see if our model was overfitting or not. Graphs were helpful in visualizing
whether or not our model is improving, or if it is overfitting or underfitting.
After we’ve trained our first well-performing model we’ve decided to use weighted average
ensemble method to further increase the accuracy. We used a similar procedure and trained 1
additional EffcientNet_v2_L model as well 2 ViT_H14 Facebook Swag Models, each with slight
variations in the unfreezing methodology to introduce stochasticity. We then evaluated individual
models separately and used their test accuracy as their weight in the ensemble model.
## Environment:
Here are the libraries that we used for our model:
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
import torchvision
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
import numpy as np
from tools.data_prep import create_dataloaders
from tools.engine import train
from tools.engine import sam_train
from tools.test_tracking import create_writer
from tools.sam import SAM
## Usage:
1. Make sure the dataset directory is structured as follows: The train folder should have 100
classes, each with 10 images. The test folder has 1000 random images.
train/
- 0/
- 0.jpg
- 1.jpg
...
- 9.jpg
- 1/
....
- 99/
...
test/
- 0.jpg
- 1.jpg
...
- 999.jpg
2. Ensure all dependencies are installed and the python version is at least 3.10. Also ensure
files are in the same directory as the models
3. To run the models, open using Jupyter Notebook and run all cells.
4. The training files are Final_3_2.ipynb, final_sam_model.ipynb, ViT.ipynbm
ViT_copy.ipynb. The Validation & Test file is ensemble.ipynb
5. Output will be saved in a folder called “predictions.csv”
