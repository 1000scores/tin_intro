import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
import os
from Tokenizers import *
from common import *
from TinyImageNet import TinyImageNet
from VisualTransformer import VisualTranformer



def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        #print(out.size())
        return out



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.nn1 = nn.Linear(dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)
        self.af1 = nn.GELU()
        self.do1 = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias, std = 1e-6)
        self.do2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.nn1(x)
        x = self.af1(x)
        x = self.do1(x)
        x = self.nn2(x)
        x = self.do2(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias = True) # Wq,Wk,Wv for each vector, thats why *3
        torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        torch.nn.init.zeros_(self.to_qkv.bias)
        
        self.nn1 = nn.Linear(dim, dim)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.zeros_(self.nn1.bias)        
        self.do1 = nn.Dropout(dropout)
        

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x) #gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1) #follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v) #product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out =  self.nn1(out)
        out = self.do1(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads = heads, dropout = dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attention, mlp in self.layers:
            x = attention(x, mask = mask) # go to attention
            x = mlp(x) #go to MLP_Block
        return x
     

class ViTResNet18(nn.Module):

    def __init__(self, block, num_blocks, batch_size_train,  num_classes=10, dim = 128, num_tokens = 8,
                 mlp_dim = 256, heads = 8, depth = 6, emb_dropout = 0.1, dropout= 0.1, tokenizers_type = ['filter', 'filter']):
        super(ViTResNet18, self).__init__()

        self.device = torch.device('cuda')
        self.batch_size_train = batch_size_train

        self.in_planes = 64
        self.L = num_tokens
        self.cT = dim
        
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) #8x8 feature maps (64 in total)
        self.apply(_weights_init)
        
        
        # Tokenization
        '''self.token_wA = nn.Parameter(torch.empty(self.batch_size_train, self.L, 256), requires_grad = True) #Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(self.batch_size_train, 256, self.cT), requires_grad = True) #Tokenization parameters
        torch.nn.init.xavier_uniform_(self.token_wV)'''
        #self.tokenizer = FilterBasedTokenizer(batch_size_train)  
        self.tokenizers_type = tokenizers_type  
        self.visual_transformer1 = VisualTranformer(batch_size_train, num_tokens, dim,
                                                     emb_dropout, depth, heads, mlp_dim, dropout, last = False, tokenizer_type=tokenizers_type[0])   
        self.visual_transformer2 = VisualTranformer(batch_size_train, num_tokens, dim,
                                                     emb_dropout, depth, heads, mlp_dim, dropout, last = True, tokenizer_type=tokenizers_type[1])      
        
        '''self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std = .02) # initialized based on the paper

        #self.patch_conv= nn.Conv2d(64,dim, self.patch_size, stride = self.patch_size) 

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) #initialized based on the paper
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()'''
        self.avgpool = nn.AdaptiveAvgPool2d((256, 1))
        self.nn1 = nn.Linear(256, num_classes)  # if finetuning, just use a linear layer without further hidden layers (paper)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std = 1e-6)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
  
        return nn.Sequential(*layers)
    
    
        
    def forward(self, img, mask = None):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)  
        x = self.layer3(x)
        # x.shape = [100, 256, 14, 14]
        x = rearrange(x, 'b c h w -> b (h w) c')
        # x.shape = [100, 196, 256]
        T, x = self.visual_transformer1(x)
        # x.shape = [100, 17, 128]
        if self.tokenizers_type[1] == 'recurrent':
            T, x = self.visual_transformer2(x, T_in=T)
        else:
            T, x = self.visual_transformer2(x)
        
        '''x = rearrange(x, 'b c h w -> b (h w) c') # 64 vectors each with 64 points. These are the sequences or word vecotrs like in NLP

        #Tokenization
        
        wa = rearrange(self.token_wA, 'b h w -> b w h') #Transpose
        A = torch.einsum('bij,bjk->bik', x, wa) 
        A = rearrange(A, 'b h w -> b w h') #Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)       
        T = torch.einsum('bij,bjk->bik', A, VV)  '''
        #print(T.size())
        # self.cls_token.shape = [1, 1, 128]
        '''cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        # x.shape = [100, 256, 14, 14]
        # T.shape = [100, 16, 128]
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask) #main game
        x = self.to_cls_token(x[:, 0])'''    
        #x = self.to_cls_token(x[:, 0])         
        #x = self.nn1(x)
        T = rearrange(T, 'b h w -> b w h') 
        T = self.avgpool(T)
        T = torch.flatten(T, 1)
        T = self.nn1(T)
        
        return T


if __name__ == '__main__':
    PATH_TO_IMAGE_NET = "./data/tiny-imagenet-200"
    BATCH_SIZE_TRAIN = 100
    BATCH_SIZE_VAL = 100
    device = torch.device("cuda")

    transform_train = torchvision.transforms.Compose(
        [torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
        torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = TinyImageNet(PATH_TO_IMAGE_NET, split='train', transform=transform_train)
    val_dataset = TinyImageNet(PATH_TO_IMAGE_NET, split='val', transform=transform_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False)

    model = ViTResNet18(BasicBlock, [2, 2, 2], BATCH_SIZE_TRAIN, num_classes=200, num_tokens=16, tokenizers_type=['filter', 'recurrent']).to(device)
    EPOCHS = 1
    check_on_dataset(model, train_loader, val_loader, EPOCHS, "TinyImageNet", "kek")