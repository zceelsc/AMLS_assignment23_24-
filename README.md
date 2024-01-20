# AMLS_assignment23_24-
all imports:

import subprocess

from tqdm import tqdm
import numpy as np
import torch, medmnist, sys
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from medmnist import INFO, Evaluator

In main.py, locationa = r'C:\Users\user\AMLS_assignment23_24-\A'  # Change this to correct file location
locationb = r'C:\Users\user\AMLS_assignment23_24-\B'  # Change this to correct file location
dataset_location = r'C:\Users\user\AMLS_assignment23_24-\Datasets'  # Change this to the desired dataset location
To run the corrsponding task A(binary) or B(classification), run main.py and it should let you input a or b. This will run the corresponding code.


