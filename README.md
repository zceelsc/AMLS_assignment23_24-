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

In main.py, location=r'C:\Users\user\AMLS_assignment23_24-\Datasets'  #change this location to the datasets folder
