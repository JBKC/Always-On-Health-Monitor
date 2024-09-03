'''
Adaptive linear model architecture
Takes in accelerometer data as input into a CNN
'''

import numpy as np
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

class AdaptiveLinearModel(nn.Module):
    def __init__(self, config):
        super().__init__()

