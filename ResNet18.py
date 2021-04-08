import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init
from common import *
import os




class ResNet18():

    def __init__(self, batch_size_train):
        self.batch_size_train = batch_size_train
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)

    def train(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)

        start_epoch = 1
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print("Loaded model's checkpoint")

        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9,weight_decay=1e-4)
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[35,48],gamma = 0.1)

        train_loss_history, test_loss_history = [], []
        last_epoch = 0
        self.train()
        for epoch in range(start_epoch, epochs + 1):
            last_epoch = epoch
            print('Epoch:', epoch)
            start_time = time.time()
            self.__train_epoch(optimizer, train_loader, train_loss_history)
            print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
            train_accuracy, test_accuracy = self.__evaluate(train_loader, train_loss_history, test_loader, test_loss_history)

            serialize_metrics(dataset_name, 'ViTRes', epoch, train_accuracy, test_accuracy)

            if epoch % 10 == 0:
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, path)
                print("Saved model's checkpoint")

        print('Execution time')