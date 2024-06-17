# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:14:19 2021

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
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
import numpy as np
import torch.backends.cudnn as cudnn; cudnn.benchmark = True




class EEGDataset:
    
    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        
        self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[20:460,:]

        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label





class Splitter:
    
    
    def __init__(self, dataset, split_path='E:\\eeg\\imagenet_dataset_eeg\\datasets\\splits_by_image.pth', split_num=0, split_name="train"):
        
        self.dataset = dataset
        
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        
        self.size = len(self.split_idx)

    
    def __len__(self):
        return self.size

    
    def __getitem__(self, i):
        
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label



class Model(nn.Module):

    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128):
        
        super().__init__()
        
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size

        
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Linear(lstm_size, output_size)
        self.classifier = nn.Linear(output_size,40)
        
    def forward(self, x):
        
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))

        
        x = self.lstm(x, lstm_init)[0][:,-1,:]
        
        
        x = F.relu(self.output(x))
        x = self.classifier((x))
        return x
    

for dataSet in ['5_95_','14_70_','55_95_']: 
    dataset = EEGDataset('E:\\eeg\\imagenet_dataset_eeg\\datasets\\eeg_' + dataSet + 'std.pth')    
    
    splits_num = 0      
    batch_size=16
    
    loaders = {split: DataLoader(Splitter(dataset,  split_num = splits_num, split_name = split), batch_size = batch_size, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}
    del dataset

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    
    print(model) 
    
        
    # Setup CUDA
    
    model.cuda()
    print("Copied to CUDA")
    
    # Start training
    for epoch in range(1, 201):
        # Initialize loss/accuracy variables
        losses = {"train": 0, "val": 0, "test": 0}
        accuracies = {"train": 0, "val": 0, "test": 0}
        counts = {"train": 0, "val": 0, "test": 0}
        
        print('Epoch : ' + str(epoch))
        # Process each split
        for split in ("train", "val", "test"):
            # Set network mode
            print(split)
            if split == "train":
                model.train()
                torch.set_grad_enabled(True)
            else:
                model.eval()
                torch.set_grad_enabled(False)
            
            for i, (input, target) in enumerate(Bar(loaders[split])):
                
                
                #print('banch : '+ str(i)+' ')
                
                input = input.cuda()
                target = target.cuda()
                # Forward
                output = model(input)
                loss = F.cross_entropy(output, target)
                losses[split] += loss.item()
                # Compute accuracy
                _,pred = output.data.max(1)
                correct = pred.eq(target.data).sum().item()
                accuracy = correct/input.data.size(0)
                accuracies[split] += accuracy
                counts[split] += 1
                # Backward and optimize
                if split == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        TrL,TrA,VL,VA,TeL,TeA=  losses["train"]/counts["train"],accuracies["train"]/counts["train"],losses["val"]/counts["val"],accuracies["val"]/counts["val"],losses["test"]/counts["test"],accuracies["test"]/counts["test"]
        
        for_print="Epoch "+str(epoch )+": TrL= "+str(TrL)+", TrA="+str(TrA)+", VL="+str(VL)+", VA="+str(VA)+", TeL="+str(TeL)+", TeA="+str(TeA)
        #print(for_print)
        
        with open('.\\encoder_model\\metrics\\'+dataSet+'\\'+str(epoch)+'.txt', 'w') as f:
            print(for_print, file=f)  
                                    
                                                                                                                                                                                                           
        if (epoch % 5 == 0) or epoch == 1:
            torch.save(model.state_dict(), '.\\encoder_model\\models\\'+dataSet+'\\encoder_'+str(epoch)+'.pth')
            