# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 10:40:24 2021

@author: nikolaos damianos
"""

# from skimage.measure import compare_ssim
# import torch.nn as nn

import imutils
import glob
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import csv
import cv2
import numpy as np
import random

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Generator(nn.Module):
    
    def __init__(self,noize_size,features_eeg):
        super(Generator,self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noize_size+features_eeg,512,4,1,0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512,256,4,2,1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,3,4,2,1,bias=False),
            nn.Tanh()
            )
        
    def forward(self,input,eeg_featur):
        output_plus_eeg = torch.cat((input,eeg_featur),1)
        output = self.main(output_plus_eeg)
        return output
    
    
G = Generator(100,128)
G.load_state_dict(torch.load("Gen/genarator_epoch_150.pth"))
G.eval()

folder = './for_sim/'

num_for_classes = 16 #eikones gia ka83 klassi
num_of_classes = 40

classes = []
for d in glob.glob('.\\data\\64_64_2\\*'):
            classes.append(d.split('\\')[-1])
            
labels_file = 'unique.csv'

with open(labels_file, mode='r') as infile:
    reader = csv.reader(infile)
    
    label_num = {int(rows[0]):rows[1] for rows in reader}
    
    
from numpy import genfromtxt

Data = genfromtxt('Relu.csv', delimiter=',')

labels = genfromtxt('Label.csv', delimiter=',').astype('int')

indexes_for_every_class = []
for i in range(40):
    f = labels == i
    result = np.where(f)
    
    gg = np.random.choice(result[0], size=num_for_classes)
    indexes_for_every_class.append(gg)
    
from PIL import Image
#with open(path, 'rb') as f:
#            img = Image.open(f)
#            return img.convert('RGB')


from skimage import data, img_as_float
# from pytorch_ssim import ssim



##############

from skimage import measure
from skimage.metrics import structural_similarity
# import imutils
# import cv2
# import scipy.misc
from numpy import savetxt


def ssim(Class):
#    misc.toimage(imf)
    true = '.\\for_sim\\true\\'+str(Class)+'\\*'
    imgsT = glob.glob(true)
    
    ImgsT = []
    for img in imgsT:
        
        img = cv2.imread(img, 1)
            
        ImgsT.append( cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))            
    
    false = '.\\for_sim\\fake\\'+str(Class)+'\\*'
    imgsF = glob.glob(false)
    
    
    ImgsF = []
    for img in imgsF:
        
        img = cv2.imread(img, 1)
            
        ImgsF.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    
#    ff = cv2.imwrite('color_img.jpg',img1)
#    tt = cv2.imwrite('color_img.jpg',img2)
#    imageA = imf
#    imageB = imt
    
#    grayA = cv2.cvtColor(ff, cv2.COLOR_BGR2GRAY)
#    grayB = cv2.cvtColor(tt, cv2.COLOR_BGR2GRAY)
    Diffspath = '.\\for_sim\\diffs\\'+str(Class)+'\\'
    
    scores = []
    
    for ii in range(16):
        
    
    
        (score, diff) = structural_similarity(ImgsT[ii], ImgsF[ii], full=True)
        scores.append(score)
        diff = (diff * 255).astype("uint8")
        cv2.imwrite(Diffspath+'diffs_'+str(ii)+'.jpg',diff)
    
    
    
    scores = np.array(scores)
    Metricspath = '.\\for_sim\\metric\\'+str(Class)+'\\'
    
    
    Csv_file_name = Metricspath + 'scores.csv'
    
    
    savetxt(Csv_file_name, scores, delimiter=',')
    
    
    for iii in range(16):
        print(scores[iii])
    
    Csv_file_name = Metricspath + 'mo.txt'
    
    Mo_file = open(Csv_file_name,"w")
    
    Mo_file.write(str(scores.sum()/16))
    Mo_file.close()
#    savetxt(Csv_file_name, scores.sum()/16)
#    print(sum(scores)/16)
    return str(scores.sum()/16)

#############

def MSE(images1,images2):
    mse=[]
    for i in range(len(images1)):
        err = np.sum((images1[i].astype("float") - images2[i].astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        mse.append(err)
    
    return err
    

M__O = [] 
  
for i in range(40):
    nz = Variable(torch.randn(num_for_classes,100,1,1))
    eegs=torch.reshape(torch.from_numpy(Data[indexes_for_every_class[i]]),(num_for_classes,128,1,1)).type(torch.float32)
    
    path_of_image = '.\\data\\64_64_2\\'+label_num[i]+'\\*'
    images_True = glob.glob(path_of_image)
    images_for_metrics = np.random.choice(images_True, size=num_for_classes)
    images_True = []
    for img in images_for_metrics:
        with open(img, 'rb') as f:
            img = Image.open(f)
            
            images_True.append(np.array(img_as_float(img)).T)
            
    images_True=torch.from_numpy(np.array(images_True))        
    images = G(nz,eegs)
    
    imf = torch.reshape(images[0],(1,64,64,3))
    imt = torch.reshape(images_True[0],(1,64,64,3))
    
    #imf = Variable(images)
#    imt = Variable(images_True)
#    ssim = ssim(imt, imf,window_size=(int(10)))
    
#    for img in range(num_for_classes):
#        
#        ssim = ssim(torch.as_tensor(images_True), images)
        #(score, diff) = compare_ssim(images_True[img].detach().numpy(), images[img].T.detach().numpy(), multichannel=True)
    num__ = 0
    for imgg in images.data:
        
        vutils.save_image(imgg,'./for_sim/fake/'+str(i)+'/'+str(num__)+'.png',normalize = True)
        num__+=1
    
    num__ = 0
    for imgg in images_True.data:
        
        vutils.save_image(imgg,'./for_sim/true/'+str(i)+'/'+str(num__)+'.png',normalize = True)
        num__+=1
    
    vutils.save_image(images.data,'./for_sim/fake/'+str(i)+'/allImages.png',normalize = True)
    vutils.save_image(images_True.data,'./for_sim/true/'+str(i)+'/allImages.png',normalize = True)
    M__O.append(ssim(i))
# ssim(img, img_noise, data_range=img_noise.max() - img_noise.min())


###################################################################################################
###################################################################################################
###################################################################################################
def find_class(cla,dataset):
    num = 0 
    for i in enumerate(dataset.labels):
       if(num == cla):
           return i[1]    
       num+=1
       
       
import Dataset40_class as dt    
import matplotlib.pyplot as plt

data_url = 'E:\\eeg\\imagenet_dataset_eeg\\datasets\\eeg_14_70_std.pth'
dataset = dt.EEGDataset(data_url)



classes = [i for i in range(40)] 

indexes = [] 






while len(classes) > 0 : 
    clas = classes.pop()
    
    print(clas)
    
    for i in range(len(dataset)):
        if dataset[i][1] == clas:
            indexes.append(i)
            break


classes = [i for i in range(40)] 
random_indexes = []


while len(classes) > 0 : 
    clas = classes.pop()
    
    print(clas)
    
    gg = True
    while gg:
        ind = random.randint(0, len(dataset)-1)
        if dataset[ind][1] == clas:
            random_indexes.append(ind)
            gg = False
    
        


for i in indexes:
    
    print(i)
    plt.title(find_class(dataset[i][1],dataset))
    plt.plot(dataset[i][0])
    plt.savefig('./for_sim/'+str(dataset[i][1])+'.png')
    

for i in random_indexes:
    
    print(i)
    plt.title(find_class(dataset[i][1],dataset))
    plt.plot(dataset[i][0])
    plt.savefig('./for_sim/random_'+str(dataset[i][1])+'.png')
    
    



Data = genfromtxt('Relu.csv', delimiter=',')
plt.plot(Data[1])
labels = genfromtxt('Label.csv', delimiter=',').astype('int')


classes = [i for i in range(40)] 
random_indexes = []

while len(classes) > 0 : 
    clas = classes.pop()
    
    print(clas)
    
    for i in range(len(Data)):
        if labels[i] == clas:
            random_indexes.append(i)
            break





for i in random_indexes:
    
    print(i)
    plt.title(find_class(labels[i],dataset))
    plt.plot(Data[i])
    plt.savefig('./for_sim/From_Relu'+str(labels[i])+'.png')


del dataset