# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 20:14:19 2021

@author: nikolaos damianos
"""

num_classes=40





import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
import numpy as np

class EEGDataset:
    
    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        
        self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        for i in enumerate(self.labels):
            print(i)
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
    
    def get_images(self):
        return self.images
    
    def get_data(self):
        return self.data
    
    def get_labels(self):
        return self.labels
    def get_targets(self):
        labels=[]
        for i in self.data:
            labels.append(i["label"])
        
        return labels#self.data[:]["label"]
    def get_eegs(self):
        #eeg = self.data[0]["eeg"].float()
        eegs = torch.zeros((self.size,128,21,21))
        
        for i in range(self.size):
            eegs[i,:,:,:]=torch.reshape(self.data[i]["eeg"][:,19:460].float(),(128,21,21))
        
        #eeg = eeg[:,:,19:460]
        return eegs
    
class Splitter:

    def __init__(self, dataset, split_path='E:\\eeg\\imagenet_dataset_eeg\\datasets\\splits_by_image.pth', split_num= 0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label
    
def get_all_labels():
    data = EEGDataset("E:\\eeg\\imagenet_dataset_eeg\\datasets\\eeg_14_70_std.pth")
    labels = []
    for i,(_,label) in enumerate(data):
        labels.append(label)
    
    del data   
    return labels

def get_labels_unique():
    data = EEGDataset("E:\\eeg\\imagenet_dataset_eeg\\datasets\\eeg_14_70_std.pth")
    labels = data.get_labels()
    
    lbs = []
    for i,lb in enumerate(labels):
        lbs.append([i,lb])
    lbs = np.array(lbs)
    print(lbs)
    np.savetxt('unique.csv', lbs, delimiter=',',fmt='%s')
    del data  
    return labels


class EEGtoImageDataset:
    
    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        
        self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.size = len(self.data)
        self.images = self.get_eegs()
#        for i in enumerate(self.labels):
#            print(i)
        # Compute size
        

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.images[i]
        
       
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label
    
    def get_images(self):
        return self.images
    
    def get_data(self):
        return self.data
    
    def get_labels(self):
        return self.labels
    def get_targets(self):
        labels=[]
        for i in self.data:
            labels.append(i["label"])
        
        return labels#self.data[:]["label"]
    def get_eegs(self):
        #eeg = self.data[0]["eeg"].float()
        eegs = torch.zeros((self.size,128,21,21))
        
        for i in range(self.size):
            eegs[i,:,:,:]=torch.reshape(self.data[i]["eeg"][:,19:460].float(),(128,21,21))
        
        #eeg = eeg[:,:,19:460]
        return eegs