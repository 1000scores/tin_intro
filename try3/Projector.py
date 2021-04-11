import PIL
import time
import torch
from torch.functional import Tensor
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
import os
from Tokenizers import FilterBasedTokenizer
from common import *
from TinyImageNet import TinyImageNet


class Projector(nn.Module):

    def __init__(self, batch_size, token_channels: int):
        super().__init__()

        self.token_channels = token_channels
        self.batch_size = batch_size

        self.Wq = nn.Parameter(torch.empty(self.batch_size, self.token_channels, self.token_channels), requires_grad = True) #Tokenization parameters
        torch.nn.init.xavier_uniform_(self.Wq)

        # self.token_wV = nn.Parameter(torch.empty(self.batch_size_train, 256, self.cT), requires_grad = True) #Tokenization parameters
        self.Wk = nn.Parameter(torch.empty(self.batch_size, self.token_channels, self.token_channels), requires_grad = True)
        torch.nn.init.xavier_uniform_(self.Wk)    

    
    def forward(self, x_in: Tensor, T: Tensor) -> Tensor:
        # x_in.shape = [100, 128]
        # x_out.shape = [100, 128]
        # T.shape = [100, 16, 128]

        A = x_in.matmul(self.Wq)
        B = T.matmul(self.Wk)
        B = rearrange(B, 'b h w -> b w h') #Transpose

        C = A.matmul(B)
        C = C.softmax(dim=-1)
        C = C.matmul(T)
        x_out = x_in + C

        return x_out

