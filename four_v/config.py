# -*- coding:utf-8 -*-
import os
from torchvision import models
from numpy import random
import time, pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np



U_LEARNING_RATE=3e-5
NN =8

BATCH_SIZE =20
NUM_WORKERS = 8
NUM_EPOCHS = 200
SIZE2 =(256,256)
SIZE3 =(350,350)
D_LEARNING_RATE =1e-4

ww =(5,50,1)

IMG_SIZE = (256, 256)
LABEL_SIZE = (256, 256)


