
import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
import os
from torch import Tensor


class FilterBasedTokenizer(nn.Module):

    def __init__(self, batch_size, num_of_tokens = 16, tokens_channels = 128):
        super().__init__()
        self.batch_size = batch_size
        self.L = num_of_tokens
        self.tokens_channels = tokens_channels

        self.token_wA = nn.Parameter(torch.empty(self.batch_size, self.L, 256), requires_grad = True) #Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wA)

        # self.token_wV = nn.Parameter(torch.empty(self.batch_size_train, 256, self.cT), requires_grad = True) #Tokenization parameters
        self.token_wV = nn.Parameter(torch.empty(self.batch_size, 256, self.tokens_channels), requires_grad = True)
        torch.nn.init.xavier_uniform_(self.token_wV)      


    def forward(self, X: Tensor) -> Tensor:
         # 64 vectors each with 64 points. These are the sequences or word vecotrs like in NLP
        #Tokenization 
        wa = rearrange(self.token_wA, 'b h w -> b w h') #Transpose
        A = torch.einsum('bij,bjk->bik', X, wa) 
        A = rearrange(A, 'b h w -> b w h') #Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', X, self.token_wV)       
        T = torch.einsum('bij,bjk->bik', A, VV)  
        
        return T
