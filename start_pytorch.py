# -*- coding: utf-8 -*-
# @Time : 2020/4/28 11:04
# @Author : zwd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

train_dataset = datasets.MNIST(root= '../dataset/mist',
                               train= True,
                               transform = transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root= '../dataset/mist',
                               train= False,
                               transform = transforms.ToTensor(),
                               download=True)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=32,
                          shuffle=False)
print(train_loader)