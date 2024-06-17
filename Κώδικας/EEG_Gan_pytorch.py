# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 18:06:21 2021

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
#from pathlib2 import Path
import glob 
from PIL import Image





class EEG_Image_Dataset(Dataset):
    def __init__(self, image_path,eegs,labels, transform=None):


        super(EEG_Image_Dataset, self).__init__()

        self.path = image_path
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(image_path)
        self.samples = self.make_dataset( self.class_to_idx)
        self.targets = [s[1] for s in self.samples]
        self.eegs = eegs
        
        print(eegs.shape)
        
        self.eegs ,_= self.Data_change_pos(labels,self.targets)
        self.mean_eegs = self.mean_eeg_all()
    def _find_classes(self, dir):
 

        classes = []

        for d in glob.glob(self.path):
            classes.append(d.split('\\')[-1])

        num_class = [self.get_id_of_label(classes[num])for num in range(len(classes))] 
        
        class_to_idx = {}
        cnt = 0
        for i in classes:
            class_to_idx[i] = num_class[cnt]
            cnt+=1

        return classes, class_to_idx

    def get_id_of_label(self,Name):
        lbs = []
        with open('unique.csv', 'r') as file:
            reader = csv.reader(file)
    
            for row in reader:
                lbs.append([int(row[0]),row[1]])
            
        for i,name in lbs:
            if name == Name :
                return i
              
    def _get_target(self, file_path):
        
        target_class=file_path.split('\\')[0]
        return target_class

    def make_dataset(self, class_to_idx):
        

        images = []
        for d in glob.glob(self.path):
            

            for img in glob.glob(d+"\\*.jpg"):
                target = self._get_target(d.split('\\')[-1])
                item = (img, class_to_idx[target])
                images.append(item)

        return images

    def get_class_dict(self):
        
        return self.class_to_idx

    def __getitem__(self, index):
        """Returns tuple: (tensor, int) where target is class_index of
        target_class.
        
        Args:
            idx: (int) Index.
        """

        Path, target = self.samples[index]
        sample = self.default_loader(Path)
        sample = self.transform(sample)

        return sample, target , self.mean_eegs[index]#self.eegs[index]

    def __len__(self):
        return len(self.samples)
    
    def pil_loader(self,path):
        
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def accimage_loader(self,path):
        import accimage
        try:
            return accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return self.pil_loader(path)
   
    
    def default_loader(self,path):
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return self.accimage_loader(path)
        else:
            return self.pil_loader(path)

    def Data_change_pos(self,Labels,from_lbs):
        
        s_data = np.zeros(self.eegs.shape)
        f_lbs=np.array(from_lbs)
        for i in range(40):
            c = f_lbs == i
            
            s_data[c] = self.eegs[Labels == i]
        
        
        return s_data , f_lbs

    def mean_eeg(self):
        s_data = np.zeros(self.eegs.shape)
        
        data_sizes_per_class = []
        images_per_class = []
        ttt = np.array(self.targets)
        for u in range(40):
            data_sizes_per_class.append(np.sum(ttt == u))
            
            images_per_class.append( int(data_sizes_per_class[u]/6))
        
        data_sizes_per_class = np.array(data_sizes_per_class)
        previus = 0
        images_per_class = np.array(images_per_class)
        
        for i in range(40):
            helper_array = np.array(self.eegs)[ttt==i]
            
            for n in range(0,images_per_class[i]):

                indexes=np.array([ind for ind in range(previus+n,np.sum(data_sizes_per_class[:i+1]),images_per_class[i])])
           
                s_data[indexes]=np.mean(helper_array[n:data_sizes_per_class[i]:images_per_class[i]],axis=0)
            previus += data_sizes_per_class[i]
            
         
        return s_data
            
    
    
    
    def mean_eeg_all(self):
        s_data = np.zeros(self.eegs.shape)
        
        data_sizes_per_class = []
        
        ttt = np.array(self.targets)
        for u in range(40):
            data_sizes_per_class.append(np.sum(ttt == u))
            
           
        
        data_sizes_per_class = np.array(data_sizes_per_class)
        previus = 0
        
        
        for i in range(40):
            helper_array = np.array(self.eegs)[ttt==i]
             
            #indexes=np.array([ind for ind in range(previus,np.sum(data_sizes_per_class[:i+1]))])
            
            s_data[previus:np.sum(data_sizes_per_class[:i+1])]=np.mean(helper_array,axis=0)
            
            previus += data_sizes_per_class[i]
            
         
        return s_data
            
#def get_name_of_label(num):
#    for i,name in enumerate(labels):
#        if num == i :
#            return name

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
#                nn.Conv2d(256,512,4,2,1,bias=False),
#                nn.BatchNorm2d(512),
#                nn.LeakyReLU(0.2,inplace=True),
#                nn.Conv2d(512,1024,4,1,0,bias=False),
#                nn.Flatten()
                )
        self.eeg_add = nn.Sequential(nn.Conv2d(256+2,512,4,2,1,bias=False),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2,inplace=True),
                                  nn.Conv2d(512,1024,4,1,0,bias=False),
                                  nn.Flatten()
                                  )
        self.output_layers=  nn.Sequential(
                nn.Linear(1024,1),
#                nn.Linear(512,1),
                nn.Sigmoid())
        
        
                
    def forward(self,input,eeg_features):
        output = self.main(input)
#        print(output.size())
        
        
        
        batch_size = eeg_features.size()[0]
        rr = torch.reshape(eeg_features,(batch_size,2,8,8))
        output_plus_eeg = torch.cat((output,rr),1)
#        print(output_plus_eeg.size())
        
        output = self.eeg_add(output_plus_eeg)
#        print(output.size())
        
#        output_plus_eeg = torch.cat((output,eeg_features),1)
        output=self.output_layers(output)
        return output.view(-1)
    
    

G = Generator(100,128).cuda()
weights_init(G)

D = Discriminator().cuda()
weights_init(D)


from numpy import genfromtxt
import numpy as np
Data = genfromtxt('Relu.csv', delimiter=',')#dt40.EEGDataset("E:\\eeg\\imagenet_dataset_eeg\\datasets\\eeg_14_70_std.pth")

labels = genfromtxt('Label.csv', delimiter=',').astype('int')


#gg = np.zeros((300*11,Data.shape[1])) 
#new_labels = [get_id_of_label(i.split('\\')[-1]) for i in glob.glob('.\\data\\64_64_2\\*')]
#lbss = np.zeros(300*11)
#prv = 0
#for i in new_labels:
#    gg[prv:prv+300] = Data[labels==i]
#    lbss[prv:prv+300] = i
#    prv+=300

#Data = Data[labels in []]
#def NLLLoss(logs, targets):
#    out = torch.zeros_like(targets, dtype=torch.float)
#    for i in range(len(targets)):
#        out[i] = logs[i][targets[i]]
#    return -out.sum()/len(out)


#img_dataset = EEG_Image_Dataset('.\\data\\64_64_2\\*',eegs=Data,labels=labels,transform=transforms.ToTensor())
img_dataset = EEG_Image_Dataset('.\\data\\64_64\\*',eegs=Data,labels=labels,transform=transforms.ToTensor()) #
print(img_dataset.class_to_idx)

img_dataset2 =dset.ImageFolder('.\\data\\Images2\\',transform=transforms.ToTensor())




criterion = nn.BCELoss().cuda()
criterion_D = nn.NLLLoss().cuda()
Disc_optimazer = optim.Adam(D.parameters(),lr=0.0001 ,betas=(0.5,0.999))
Gen__optimazer = optim.Adam(G.parameters(),lr=0.001 ,betas=(0.5,0.999))


dataloader2 = DataLoader(img_dataset2,shuffle=True, batch_size=16)
dataloader = DataLoader(img_dataset,shuffle=True, batch_size=16)

#                                                  D  B    C
#nll einai to -log pou epistrefei to neuroniko px [0 0.73 0.27] b= bird , d = dog , c = cat kai to target einai [0 1 0] 
# opote to loss einai -log(0.73) = 0.314 


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




#ep = 20
#for epoch in range(ep):
#    for i , (data,_) in enumerate(dataloader2,0):
#        D.zero_grad()
#        
#        real_images = data
#        Input = Variable(real_images)
#        Input_size=Input.size()[0]
#        
#        
#        eegs = torch.zeros(Input_size,128).cuda()#eegs.cuda(async = True)#Input_size,128,1,1)
#        eegs_for_G = torch.reshape(eegs,(Input_size,128,1,1))
#        
#        
#
#        
#        target = Variable(torch.ones(Input_size)).cuda()#,dtype=torch.long)).cuda(async = True)
#        Input=Input.cuda()
#        output = D(Input.float(),eegs.float())
#
#        errD_real = criterion(output,target)#criterion(output,target)
#
#        
#        nz = Variable(torch.randn(Input_size,100,1,1)).cuda()
#        fake_images = G(nz.float(),eegs_for_G.float())
#        
#        target_f = Variable(torch.zeros(Input_size)).cuda()#Variable(torch.zeros(Input_size))
#        
#        output_f = D(fake_images.detach().float(),eegs.float())
#        
#
#        
#        errD_fake =  criterion(output_f,target_f)
#        
#
#        errD =errD_real+errD_fake
#        errD.backward()
#        
#        Disc_optimazer.step()
#        
#        G.zero_grad()
#        
#        fake = G(nz.float(),eegs_for_G.float())
#        target_G =Variable(torch.ones(Input_size)).cuda() #Variable(torch.ones(Input_size))
#        output_G = D(fake.float(),eegs.float())
#        
#        errG=criterion(output_G,target_G)#NLL_Loss_G(output_G.detach().cpu().data.numpy())#
#        errG.backward() 
#        Gen__optimazer.step()
#        
#        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' %(epoch,ep,i,len(dataloader2),errD.data,errG.data))
#        
##        if i % 60 == 0:
##            vutils.save_image(real_images,'%s/real_samples.png' % "./results",normalize = True)
##            
##            vutils.save_image(fake.data,'./results/false_'+str(epoch)+'_'+str(i)+'_samples.png',normalize = True)
#    if (epoch+1) % 10 == 0:
#        torch.save(G.state_dict(), '.\\Gen\\genaratorPart1_epoch_'+str(epoch+1)+'.pth')
#        torch.save(D.state_dict(), '.\\Disc\\discriminatorPart2_epoch_'+str(epoch+1)+'.pth')  





num_of_epochs=50

#part2
for epoch in range(num_of_epochs):
    for i , (data,_,eegs) in enumerate(dataloader,0):
        D.zero_grad()
        
        real_images = data
        Input = Variable(real_images)
        Input_size=Input.size()[0]
        
        
        eegs = eegs.cuda()
        eegs_for_G = torch.reshape(eegs,(Input_size,128,1,1))
        
        
        #eps=1e-7
        
        
        target = Variable(torch.ones(Input_size)).cuda()#,dtype=torch.long)).cuda(async = True)
        Input=Input.cuda()
        output = D(Input.float(),eegs.float())

        errD_real = criterion(output,target)#criterion(output,target)
#        output[output==0]= 1
#        edr = -torch.log(output+eps)
#        edr[torch.isinf(edr)] = 1
        
        nz = Variable(torch.randn(Input_size,100,1,1)).cuda()
        fake_images = G(nz.float(),eegs_for_G.float())
        
        target_f = Variable(torch.zeros(Input_size)).cuda()#Variable(torch.zeros(Input_size))
        
        output_f = D(fake_images.detach().float(),eegs.float())
        
        errD_fake =  criterion(output_f,target_f)
#        eDf = -torch.log( 1-output_f)
#        eDf[torch.isinf(eDf)] = 1
        
        output_ft = D(Input.float(),eegs.float())
#        edft=-torch.log( 1-output_ft)
#        edft[torch.isinf(edft)] = 1
        errD_fake_true = criterion(output_ft,target_f)

        errD =errD_real+(errD_fake)+(errD_fake_true)
#        errD = torch.mean(edr+eDf+edft)
        #print(NLL_Loss_D(errD_real.detach().cpu(),errD_fake.detach().cpu(),errD_fake_true.detach().cpu()))
        
        #tt=NLL_Loss_D(output.detach().cpu(),output_f.detach().cpu(),output_ft.detach().cpu())
        
        errD.backward()
        Disc_optimazer.step()
        
        G.zero_grad()
        
        fake = G(nz.float(),eegs_for_G.float())
        target_G =Variable(torch.ones(Input_size)).cuda() #Variable(torch.ones(Input_size))
        output_G = D(fake.float(),eegs.float())
#        print(output_G)
        errG=criterion(output_G,target_G)#NLL_Loss_G(output_G.detach().cpu().data.numpy())#
        #Glog=torch.log(output_G+eps)
        #Glog[torch.isinf(Glog) | torch.isnan(Glog)]=1
        #errG  = -torch.mean(Glog)
        #print(errG)
        errG.backward() 
        Gen__optimazer.step()
        
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' %(epoch,num_of_epochs,i,len(dataloader),errD.data,errG.data))
        
        if i % 32 == 0:
            vutils.save_image(real_images,'%s/real_samples.png' % "./results",normalize = True)
            
            vutils.save_image(fake.data,'./results/false_'+str(epoch)+'_'+str(i)+'_samples.png',normalize = True)
    
    if (epoch+1) % 10 == 0:
        torch.save(G.state_dict(), '.\\Gen\\genarator_epoch_'+str(epoch+1)+'.pth')
        torch.save(D.state_dict(), '.\\Disc\\discriminator_epoch_'+str(epoch+1)+'.pth')    


#from torchvision.models.inception import inception_v3


#import matplotlib.pyplot as plt
#import xml.etree.ElementTree as ET 
#
#import torch
#from torch import nn
#
#from torchvision.utils import save_image
#
#from PIL import Image
#import glob
#
#data_dir = 'C:\\Users\\nikolaos damianos\\Desktop\\eeg\\data\\Images\\*\\*'
#all_images=glob.glob(data_dir)
#
#plt.figure(figsize=(3,3))
#for i in range(100,109):
#    plt.subplot(3,3,(i-100)+1)
#    im=Image.open(all_images[i])
#    plt.imshow(im)
#    plt.axis("off")

