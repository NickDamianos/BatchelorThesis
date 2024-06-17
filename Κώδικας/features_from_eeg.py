# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 22:37:21 2021

@author: nikolaos damianos
"""

from barbar import Bar
from torch.autograd import Variable
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import Dataset40_class as dt



class Model(nn.Module):

    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size

        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)
        self.classifier = nn.Linear(output_size,40)
        
    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0]), Variable(lstm_init[1]))

        # Forward LSTM and get final state
        x = self.lstm(x  , lstm_init)[0][:,-1,:]#
        
        # Forward output
        x_relu = F.relu(self.output(x))
        x_classif = self.classifier((x_relu))
        return x , x_relu , x_classif





data_url = 'E:\\eeg\\imagenet_dataset_eeg\\datasets\\eeg_14_70_std.pth'
dataset = dt.EEGDataset(data_url) 

splits_num = 0      
batch_size=16




#images = dataset.get_images()
model_path='.\\encoder_model\\models\\14_70_\\encoder_200.pth'
model = Model().to('cuda:0')
model.load_state_dict(torch.load(model_path))
model.eval()

dataload = DataLoader(dataset, batch_size=batch_size, shuffle=False)


cnt=0
x = []
x_relu = []
labels = []

for i , (data , label) in enumerate(dataload):
    
        data = data.cuda(non_blocking = True)
        #print(label)
        #print(model.output(data))
        out,relu,_ = model(data)
        for uu in out.detach().cpu().clone().numpy():#.detach().cpu().clone().numpy():
            x.append(uu)
        
        for uu in relu.detach().cpu().clone().numpy():
            x_relu.append(uu)
       
        for uu in label.numpy():
            labels.append(uu)

x = np.array(x)
np.savetxt('Lstm.csv', x, delimiter=',')

x_relu = np.array(x_relu)
np.savetxt('Relu.csv', x_relu, delimiter=',')


ll = np.array(labels)
np.savetxt('Label.csv', ll, delimiter=',')
del dataset 