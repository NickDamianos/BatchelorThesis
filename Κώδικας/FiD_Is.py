# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 03:47:23 2022

@author: nikolaos damianos
"""


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

import glob 
from PIL import Image

import torchvision
import ignite
import logging
import matplotlib.pyplot as plt

from ignite.engine import Engine, Events
import ignite.distributed as idist

ignite.utils.manual_seed(999)
ignite.utils.setup_logger(name="ignite.distributed.auto.auto_dataloader", level=logging.WARNING)
ignite.utils.setup_logger(name="ignite.distributed.launcher.Parallel", level=logging.WARNING)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

          


def get_id_of_label(Name):
    lbs = []
    with open('unique.csv', 'r') as file:
        reader = csv.reader(file)
    
        for row in reader:
            lbs.append([int(row[0]),row[1]])
        
    
    for i,name in lbs:
        if name == Name :
            return i

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

noize_size=100
features_eeg = 128

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
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(3,64,4,2,1,bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(64,128,4,2,1,bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(128,256,4,2,1,bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2,inplace=True),

                )
        self.eeg_add = nn.Sequential(nn.Conv2d(256+2,512,4,2,1,bias=False),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2,inplace=True),
                                  nn.Conv2d(512,1024,4,1,0,bias=False),
                                  nn.Flatten()
                                  )
        self.output_layers=  nn.Sequential(
                nn.Linear(1024,1),
                nn.Sigmoid())
        
        
                
    def forward(self,input,eeg_features):
        output = self.main(input)

        
        
        
        batch_size = eeg_features.size()[0]
        rr = torch.reshape(eeg_features,(batch_size,2,8,8))
        output_plus_eeg = torch.cat((output,rr),1)

        
        output = self.eeg_add(output_plus_eeg)

        output=self.output_layers(output)
        return output.view(-1)
    
    
G = idist.auto_model(Generator(100,128))



D = idist.auto_model(Discriminator())



from numpy import genfromtxt
import numpy as np
Data = genfromtxt('Relu.csv', delimiter=',')



labels = genfromtxt('Label.csv', delimiter=',').astype('int')





image_size = 64

data_transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
from torchvision.datasets import ImageFolder

img_dataset =dset.ImageFolder('.\\data\\64_64\\',transform=transforms.ToTensor())

test_dataset = torch.utils.data.Subset(img_dataset, torch.arange(3000))


criterion = nn.BCELoss().cuda()
criterion_D = nn.NLLLoss().cuda()

fixed_noise = torch.distributions.uniform.Uniform(-1,1).sample([64, 100, 1, 1]).to(device=idist.device())#torch.randn(64, 100, 1, 1, device=idist.device())


Disc_optimazer = idist.auto_optim(
    optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
)

Gen__optimazer = idist.auto_optim(
    optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
)





dataloader = idist.auto_dataloader(
    img_dataset, 
    batch_size=16, 
    num_workers=2, 
    shuffle=True, 
    drop_last=True,
)



test_dataloader = idist.auto_dataloader(
    test_dataset, 
    batch_size=16, 
    num_workers=2, 
    shuffle=False, 
    drop_last=True,
)






from scipy import special

def NLL_Loss_G(output):
    Loss = special.xlogy(1, output)#+special.xlogy(1-target, 1-output)
    return Variable(-torch.mean(torch.tensor(Loss)),requires_grad=True).cuda()

def NLL_Loss_D(output_t,output_f,output_ft):
    #Loss_t = -special.xlogy(1, output_t)#+special.xlogy(1-target, 1-output)
    Loss_t = -torch.log( output_t)
    
    #Loss_f = -special.xlogy(1, 1-output_f)
    Loss_f = -torch.log( 1-output_f)
    
    
    #Loss_ft = -special.xlogy(1, 1-output_t)
    
    Loss_ft =-torch.log( 1-output_t)
    return torch.mean(torch.tensor(Loss_t+Loss_f+Loss_ft)).cuda().requires_grad_(True)#torch.mean(torch.tensor(Loss_t+Loss_f+Loss_ft)).cuda()


real_label = 1
fake_label = 0



def training_step(engine, data):
    # Set the models for training
    G.train()
    D.train()
    b_size = data[0].size(0)

    D.zero_grad()
    ind= random.choices(np.arange(len(eegs_train)),k=b_size)
    e=eegs_train[ind]
    ee = np.delete(eegs_train, ind)
    eegs = torch.tensor(e).float().cuda()
    
        
    real = data[0].to(idist.device())
    
    label = torch.full((b_size,), real_label, dtype=torch.float, device=idist.device())
    
    eegs = torch.reshape(eegs,(b_size,128,1,1))
    
    output1 = D(real,eegs.float()).view(-1)
    
    errD_real = criterion(output1, label)
    
    errD_real.backward()

   
    noise = torch.randn(b_size, 100, 1, 1, device=idist.device())
    
    fake = G(noise,eegs)
    label.fill_(fake_label)
    
    output2 = D(fake.detach(),eegs.float()).view(-1)
    
    errD_fake = criterion(output2, label)
    
    errD_fake.backward()
    
    errD = errD_real + errD_fake
    
    Disc_optimazer.step()

    
    G.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    
    output3 = D(fake,eegs.float()).view(-1)

    errG = criterion(output3, label)
    
    errG.backward()
    
    Gen__optimazer.step()

    return {
        "Loss_G" : errG.item(),
        "Loss_D" : errD.item(),
        "D_x": output1.mean().item(),
        "D_G_z1": output2.mean().item(),
        "D_G_z2": output3.mean().item(),
    }


trainer = Engine(training_step)

eegs_train=Data

@trainer.on(Events.STARTED)
def init_weights():
    D.apply(weights_init)
    G.apply(weights_init)


G_losses = []
D_losses = []


@trainer.on(Events.ITERATION_COMPLETED)
def store_losses(engine):
    o = engine.state.output
    G_losses.append(o["Loss_G"])
    D_losses.append(o["Loss_D"])
    
    
img_list = []


@trainer.on(Events.ITERATION_COMPLETED(every=150))
def store_images(engine):
    with torch.no_grad():
        ind= random.choices(np.arange(len(Data)),k=64)
        e=Data[ind]
        
        eegs = torch.tensor(e).float().cuda()
        eegs = torch.reshape(eegs,(64,128,1,1))
        fake = G(fixed_noise,eegs).cuda()
    img_list.append(fake)


from ignite.metrics import FID, InceptionScore

fid_metric = FID(device=idist.device())
is_metric = InceptionScore(device=idist.device(), output_transform=lambda x: x[0])



def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)

import random


def evaluation_step(engine, batch):
    with torch.no_grad():
        noise = torch.randn(16, 100, 1, 1, device=idist.device())
        G.eval()
        ind= random.choices(np.arange(len(Data)),k=16)
        e=Data[ind]
        
        eegs = torch.tensor(e).float().cuda()
        eegs = torch.reshape(eegs,(16,128,1,1))
        fake_batch = G(noise,eegs).cuda()
        fake = interpolate(fake_batch)
        real = interpolate(batch[0])
        return fake, real
    
evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")
is_metric.attach(evaluator, "is")

fid_values = []
is_values = []


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(test_dataloader,max_epochs=1)
    metrics = evaluator.state.metrics
    fid_score = metrics['fid']
    is_score = metrics['is']
    fid_values.append(fid_score)
    is_values.append(is_score)
    print(f"Epoch [{engine.state.epoch}/5] Metric Scores")
    print(f"*   FID : {fid_score:4f}")
    print(f"*    IS : {is_score:4f}")
    eegs_train=Data


from ignite.metrics import RunningAverage


RunningAverage(output_transform=lambda x: x["Loss_G"]).attach(trainer, 'Loss_G')
RunningAverage(output_transform=lambda x: x["Loss_D"]).attach(trainer, 'Loss_D')

from ignite.contrib.handlers import ProgressBar


ProgressBar().attach(trainer, metric_names=['Loss_G','Loss_D'])
ProgressBar().attach(evaluator)


def training(*args):
    trainer.run(dataloader, max_epochs=20)

if __name__ == '__main__':
    training()
    textfile = open("IS_metrics.txt", "w")
    ind = 1 
    for element in is_values:
        textfile.write("epoch : "+ str(ind) + " | "+str(element) + "\n")
        ind+=1
    textfile.close()

    ind = 1
    textfile = open("FID_metrics.txt", "w")
    for element in fid_values:
        textfile.write("epoch : "+ str(ind) + " | "+str(element) + "\n")
        ind+=1
    textfile.close()

    from torchvision.utils import save_image
    
    save_folder = "images_from_metric_evaluations\\"
    for i in range(len(img_list)):
        for y in range(img_list[i].shape[0]):
            
            imgM = img_list[i][y] #torch.Size([3,28,28]
            # img1 = img1.numpy() # TypeError: tensor or list of tensors expected, got <class 'numpy.ndarray'>
            save_image(imgM, save_folder+'img_'+str(i)+'_'+str(y)+'.png')


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()

plt.savefig('Generator_and_Discriminator_Loss_During_Training.jpg')

