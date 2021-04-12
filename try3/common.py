
import os
import json
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
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
import cv2
from torch.optim.lr_scheduler import StepLR

'''
data/dataset_name
metrics/dataset_name/net_name.txt  (epoch, train_accuracy, test_accuracy, test_precision, test_recall)
saved_nets/dataset_name/net_name.pt
graphs/dataset_name/metric/*

'''
def log_all(dataset_name: str, model: nn.Module, model_name: str, epoch, train_accuracy, test_accuracy, test_precision, test_recall):
    serialize_metrics(dataset_name, model_name, epoch, train_accuracy, test_accuracy, test_precision, test_recall)
    torch.save(model.state_dict(), f'nets/{model_name}')

def serialize_metrics(dataset_name: str, model_name: str, epoch, train_accuracy, test_accuracy):
    if not os.path.isdir('./metrics'):
        os.mkdir('metrics')
    
    if not os.path.isdir(f'./metrics/{dataset_name}'):
        os.mkdir(f'metrics/{dataset_name}')
    
    path = f'./metrics/{dataset_name}/{model_name}.json'
    metrics = dict()
    if os.path.isfile(path):
        with open(path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics['epoch'] = list()
        metrics['train_accuracy'] = list()   
        metrics['test_accuracy'] = list()
    
    metrics['epoch'].append(epoch)
    metrics['train_accuracy'].append(train_accuracy)
    metrics['test_accuracy'].append(test_accuracy)

    with open(path, 'w') as f:
        json.dump(metrics, f)


def deserialize_metrics(dataset_name: str, model_name: str):
    path = f'metrics/{dataset_name}/{model_name}.json'
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        print("Can't deserialize these metrics!")
        return None

def resize_img(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(size,size), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(image_path,img)


def __graph_compare(dataset_name: str, model1_metrics, model2_metrics, model1_name: str, model2_name: str, metric_name: str, n_epochs: int):
    plt.xlabel("epochs(%)")
    plt.ylabel(metric_name)
    plt.title(f'{dataset_name}(epochs = {n_epochs})')
    
    X1 = np.arange(1, 101)
    X2 = np.arange(1, 101)

    model1_metrics = model1_metrics[:n_epochs]
    model2_metrics = model2_metrics[:n_epochs]
    
    y1 = []
    y2 = []

    for i in X1:
        y1.append(model1_metrics[min(int((i / 100.0) * len(model1_metrics)), len(model1_metrics) - 1)])

    for i in X2:
        y2.append(model2_metrics[min(int((i / 100.0) * len(model2_metrics)), len(model2_metrics) - 1)])
    


    plt.plot(X1, y1, c='b', label=model1_name)
    plt.plot(X2, y2, c='r', label=model2_name)
    plt.legend()


    if not os.path.isdir(f'./graphs'):
        os.mkdir(f'./graphs')
    
    if not os.path.isdir(f'./graphs/{dataset_name}'):
        os.mkdir(f'./graphs/{dataset_name}')

    if not os.path.isdir(f'./graphs/{dataset_name}/{metric_name}'):
        os.mkdir(f'./graphs/{dataset_name}/{metric_name}')
    
    plt.savefig(f'graphs/{dataset_name}/{metric_name}/{metric_name}_{model1_name}vs{model2_name}')

    plt.show()


def create_graph_comparison(dataset_name: str, model1_name: str, model2_name: str):
    model1_metrics = deserialize_metrics(dataset_name, model1_name)
    model2_metrics = deserialize_metrics(dataset_name, model2_name)

    if (model1_metrics is None) or (model2_metrics is None):
        print("Can't find saved metrics")
        return
    
    __graph_compare(dataset_name, model1_metrics['train_accuracy'], model2_metrics['train_accuracy'], model1_name, model2_name,
                    "train_accuracy", min(model1_metrics['epoch'][-1], model2_metrics['epoch'][-1]))
    
    __graph_compare(dataset_name, model1_metrics['test_accuracy'], model2_metrics['test_accuracy'], model1_name, model2_name,
                    "test_accuracy", min(model1_metrics['epoch'][-1], model2_metrics['epoch'][-1]))





def __train_epoch(model, optimizer, data_loader, loss_history, device="cuda"):
    total_samples = len(data_loader.dataset)
    model.train()
    
    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = F.log_softmax(model(data), dim=1)
        output = output.to(device)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                    ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                    '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


def __get_loss_and_accuracy(model, data_loader, loss_history, device="cuda") -> float:
    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            target = target.to(device)
            data = data.to(device)
            output = F.log_softmax(model(data), dim=1)
            output = output.to(device)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)

    return 100.0 * correct_samples / total_samples

def __evaluate(model, train_data_loader, train_loss_history, test_data_loader, test_loss_history):
    model.eval()

    train_accuracy = __get_loss_and_accuracy(model, train_data_loader, train_loss_history)
    test_accuracy = __get_loss_and_accuracy(model, test_data_loader, test_loss_history)
    
    print(f'\nAverage train loss: ' + '{:.4f}'.format(train_loss_history[-1]))
    print(f'\nTrain accuracy: ' + '{:.4f}'.format(train_accuracy))
    print(f'\nAverage test loss: ' + '{:.4f}'.format(test_loss_history[-1]))
    print(f'\nTest accuracy: ' + '{:.4f}'.format(test_accuracy))

    return (train_accuracy, test_accuracy)


def check_on_dataset(model, train_loader, test_loader, epochs, dataset_name, model_name):
        if not os.path.isdir(f"./saved_nets/{dataset_name}"):
            os.mkdir(f"./saved_nets/{dataset_name}")

        path = f"saved_nets/{dataset_name}/{model_name}.pt"
        
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, nesterov=True, momentum=0.9, weight_decay=4e-5)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        start_epoch = 1
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print("Loaded model's checkpoint")

        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=.9,weight_decay=1e-4)
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[35,48],gamma = 0.1)

        train_loss_history, test_loss_history = [], []
        last_epoch = start_epoch
        model.train()
        for epoch in range(start_epoch, epochs + 1):
            last_epoch = epoch
            print('Epoch:', epoch)
            start_time = time.time()
            __train_epoch(model, optimizer, train_loader, train_loss_history)
            scheduler.step()
            print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
            if epoch % 15 == 0:
                train_accuracy, test_accuracy = __evaluate(model, train_loader, train_loss_history, test_loader, test_loss_history)
                serialize_metrics(dataset_name, model_name, epoch, train_accuracy.item(), test_accuracy.item())

            if epoch % 10 == 0:
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, path)
                print("Saved model's checkpoint")

        print('Execution time')

        torch.save({
            'epoch': last_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, path)

        