
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

    def __init__(self, batch_size, num_of_tokens = 16, tokens_channels = 256):
        super().__init__()

        self.Wa = nn.Parameter(torch.empty(batch_size, tokens_channels, num_of_tokens), requires_grad = True) #Tokenization parameters
        torch.nn.init.xavier_uniform_(self.Wa)

        # self.token_wV = nn.Parameter(torch.empty(self.batch_size_train, 256, self.cT), requires_grad = True) #Tokenization parameters
        #self.token_wV = nn.Parameter(torch.empty(self.batch_size, 256, self.tokens_channels), requires_grad = True)
        #torch.nn.init.xavier_uniform_(self.token_wV)      


    def forward(self, X: Tensor) -> Tensor:
         # 64 vectors each with 64 points. These are the sequences or word vecotrs like in NLP
        #Tokenization 
        #wa = rearrange(self.token_wA, 'b h w -> b w h') #Transpose
        #A = torch.einsum('bij,bjk->bik', X, self.Wa)
        A = X.matmul(self.Wa)
        A = A.softmax(dim=1)
        A = rearrange(A, 'b h w -> b w h') #Transpose
        

        return A.matmul(X)

        '''VV = torch.einsum('bij,bjk->bik', X, self.token_wV)       
        T = torch.einsum('bij,bjk->bik', A, VV)  
        
        return T'''

class PoolingBasedTokenizer(nn.Module):

    def __init__(self, HW = 196):
        super().__init__()
        self.pooling = nn.AvgPool1d(kernel_size=16, stride=12, padding=0)


    def forward(self, X: Tensor) -> Tensor:
        x = rearrange(X, 'b h w -> b w h') #Transpose
        x = self.pooling(x)
        x = rearrange(x, 'b h w -> b w h')
        return x

class RecurrentBasedTokenizer(nn.Module):

    def __init__(self, batch_size, tokens_channels=256, num_of_tokens=16):
        super().__init__()
        
        self.W_T_r = nn.Parameter(torch.empty(batch_size, tokens_channels, tokens_channels), requires_grad = True)
        self.Wr = nn.Parameter(torch.empty(batch_size, tokens_channels, num_of_tokens), requires_grad = True)\
    
    def forward(self, X: Tensor, T: Tensor) -> Tensor:
        # X.shape = [100, 196, 256]
        x = X.matmul(self.W_T_r)
        A = x.matmul(self.Wr)
        A = A.softmax(dim=1)
        A = rearrange(A, 'b h w -> b w h')
        return A.matmul(X)
        

