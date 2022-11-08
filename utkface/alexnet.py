import numpy as np
import torch
import torch.nn as nn


class cnnModel(nn.Module):
  def __init__(self, num_channels=1, num_classes=2):
