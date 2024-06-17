# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 23:05:02 2021

@author: nikolaos damianos
"""

from barbar import Bar
from torch.autograd import Variable
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader ,Dataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
import numpy as np
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import Dataset40_class as dt


#class Model(nn.Module):
#
#    def __init__(self, input_size=128, lstm_size=128, lstm_layers=1, output_size=128):
#        # Call parent
#        super().__init__()
#        # Define parameters
#        self.input_size = input_size
#        self.lstm_size = lstm_size
#        self.lstm_layers = lstm_layers
#        self.output_size = output_size
#
#        # Define internal modules
#        self.lstm = nn.Sequential(nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True),
#                                  nn.Dropout(0.4))
#        self.lstm2 = nn.Sequential(nn.LSTM(input_size, lstm_size, num_layers=lstm_layers),
#                                  nn.Dropout(0.4))
#        self.output = nn.Linear(lstm_size, 40)
#        
#        self.classifier = nn.Softmax(dim=1)
#        
#    def forward(self, x):
#        # Prepare LSTM initiale state
#        batch_size = x.size(0)
#        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
#        lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
#        lstm_init = (Variable(lstm_init[0]), Variable(lstm_init[1]))
#        print(lstm_init)
#        # Forward LSTM and get final state
#        x = self.lstm(x, lstm_init)[0][:,-1,:]
#        print(x)
#        # Forward output
#        x = F.relu(self.output(x))
#         
#        
#        x = self.classifier((x))
#        return x
 
class Model2(nn.Module):
    def __init__(self, input_size=128, classes=40):
        super(Model2,self).__init__()
        #128*21*21
        self.Convs = nn.Sequential(              
                nn.Conv2d(128,128,4),
                #nn.MaxPool2d(2, 2),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128,256,3),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,128,2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Flatten()
                )
        self.Dense = nn.Sequential(
                nn.Linear(28800,1024),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(1024,256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256,40)
#                nn.Tanh()
                )
    
    def forward(self, x):
        output = self.Convs(x)
#        print(output.size())
        output = self.Dense(output)
        return output
class Model(nn.Module):

    def __init__(self, input_size=128, lstm_size=128, lstm_layers=2, output_size=128):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.output_size = output_size

        # Define internal modules
        self.lstm = nn.LSTM(input_size, lstm_size, num_layers=lstm_layers, batch_first=True)
        self.output = nn.Sequential(nn.Linear(lstm_size, output_size),
                                    nn.ReLU(),
                                    nn.Dropout(0.4),
                                    nn.Linear(output_size, output_size),
                                    nn.Dropout(0.4),
                                    nn.Linear(output_size, int(output_size/2)),
                                    nn.Dropout(0.3))
        
        self.classifier =nn.Linear(int(output_size/2),40) #nn.logSoftmax(dim=1)
        
    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))

        # Forward LSTM and get final state
        x = self.lstm(x, lstm_init)[0][:,-1,:]
        
        # Forward output
#        x = F.relu(self.output(x))
        x = self.classifier((self.output(x)))
        return x   
    
data_url = 'E:\\eeg\\imagenet_dataset_eeg\\datasets\\eeg_14_70_std.pth'
#data_url = 'E:\\eeg\\imagenet_dataset_eeg\\datasets\\eeg_5_95_std.pth'
#data_url = 'E:\\eeg\\imagenet_dataset_eeg\\datasets\\eeg_55_95_std.pth'

#dataset = dt.EEGDataset(data_url) 
dataset = dt.EEGtoImageDataset(data_url) 

    




 
batch_size=16

model = Model2()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)#Adam(model.parameters(), lr = 0.001)
    
#dataload = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.cuda()
len_train= int(dataset.size*0.7)
from torch.utils.data.sampler import SubsetRandomSampler
dataset_size = len(dataset)
indices = list(range(dataset_size))
#split = int(np.floor(0.2 * dataset_size))

len_test = dataset.size-len_train
len_val = len_test - int(len_test* 0.33)
len_test = len_test - len_val

np.random.seed(42)
np.random.shuffle(indices)

train_indices, val_indices , test_indices= indices[:len_train], indices[len_train:len_train+len_val], indices[len_train+len_val:]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler =  SubsetRandomSampler(test_indices)

train_set= DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_set = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
test_set = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
#loaders = {split: DataLoader(dt.Splitter(dataset,  split_num = splits_num, split_name = split), batch_size = batch_size, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}
del dataset


loaders = {}
loaders['train'] = train_set
loaders['val'] = val_set
loaders['test'] = test_set

for epoch in range(1, 101):
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
            # Process all split batches
            for i, (input, target) in enumerate(Bar(loaders[split])):
                # Check CUDA
                
                #print('banch : '+ str(i)+' ')
                
                input = input.cuda(async = True)
                target = target.cuda(async = True)
                # Forward
                output = model(input)
                target2 = torch.zeros((batch_size,40))
                cnt = 0
                for i in target:
                    target2[cnt,i]=1
                    cnt+=1
                target2 = target2.long().cuda(async = True)
                loss =F.cross_entropy(output, target) #F.nll_loss(torch.log(output),target)
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
        
        print('\n accuracy = '+str(accuracies["train"]/counts["train"])+'\n loss = '+str(losses["train"]/counts["train"])+' \n')
        TrL,TrA,VL,VA,TeL,TeA=  losses["train"]/counts["train"],100*accuracies["train"]/counts["train"],losses["val"]/counts["val"],100*accuracies["val"]/counts["val"],losses["test"]/counts["test"],100*accuracies["test"]/counts["test"]
        # Print info at the end of the epoch
        for_print="Epoch "+str(epoch )+": TrL= "+str(TrL)+", TrA="+str(TrA)+", VL="+str(VL)+", VA="+str(VA)+", TeL="+str(TeL)+", TeA="+str(TeA)
        #print(for_print)
        
        with open('.\\encoder_model\\metrics\\14_70_\\'+str(epoch)+'.txt', 'w') as f:
            print(for_print, file=f)  
                                    
                                                                                                                                                                                                           
        if (epoch % 5 == 0) or epoch == 1:
            torch.save(model.state_dict(), '.\\encoder_model\\models\\14_70_\\class_\\'+str(epoch)+'.pth')
            
            
            
            
#CDCGAN
            
embsize = 40        


class Generator(nn.Module):
    
    def __init__(self,noize_size=100,classes=11):
        super(Generator,self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noize_size+classes,512,4,1,0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,3,4,2,1),
            nn.Tanh()
            )
        
    def forward(self,input,labels):
        output_plus_eeg = torch.cat((input,labels),1)
        output = self.main(output_plus_eeg)
        return output    

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(3,64,4,2,1,bias=False),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(64,128,3,2,1,bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(128,256,3,2,1,bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Conv2d(256,512,2,2,1,bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Flatten()
                )
#        self.eeg_add = nn.Sequential(nn.Conv2d(256,512,4,2,1,bias=False),
#                                  nn.BatchNorm2d(512),
#                                  nn.LeakyReLU(0.2,inplace=True),
#                                  nn.Flatten()
#                                  )
        self.output_layers=  nn.Sequential(
                nn.Linear(12811,1),
                
                nn.Sigmoid())
        
        
                
    def forward(self,input,labels):
        output = self.main(input)
        #print(output.size())
        
        
        
        
        output_plus = torch.cat((output,labels),1)
        #print(output_plus.size())
#        print(output_plus)
        
#        output = self.eeg_add(output_plus)
#        print(output.size())
        
#        output_plus_eeg = torch.cat((output,eeg_features),1)
        output=self.output_layers(output_plus)
        return output.view(-1)


    
    
input = torch.FloatTensor(16, 3, 64, 64)
noise = torch.FloatTensor(16, 100, 1, 1)

c_emb =torch.FloatTensor(16, embsize, 1, 1)
fixed_noise = torch.FloatTensor(16, 100, 1, 1).normal_(0, 1)

input = Variable(input)

noise = Variable(noise)
fixed_noise = Variable(fixed_noise)
c_emb = Variable(c_emb)


import glob 
from PIL import Image
import csv
from sklearn.preprocessing import LabelBinarizer
import torch.optim as optim


label_binarizer = LabelBinarizer()
label_binarizer.fit(range(11))

c_one_hot = torch.from_numpy(label_binarizer.transform([10,3]))
c_emb_data = c_one_hot.type(torch.FloatTensor)

class EEG_Image_Dataset(Dataset):
    def __init__(self, image_path, transform=None):


        super(EEG_Image_Dataset, self).__init__()

        self.path = image_path
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(image_path)
        self.samples = self.make_dataset( self.class_to_idx)
        self.targets = [s[1] for s in self.samples]
        
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

        return sample, target 

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



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

G = Generator().cuda()
weights_init(G)

D = Discriminator().cuda()
weights_init(D)

img_dataset =EEG_Image_Dataset('.\\data\\Images2\\*',transform=transforms.ToTensor())


dataloader = DataLoader(img_dataset,shuffle=True, batch_size=16)

criterion = nn.BCELoss().cuda()
Disc_optimazer = optim.Adam(D.parameters(),lr=0.0001,betas=(0.5,0.999))
Gen__optimazer = optim.Adam(G.parameters(),lr=0.0001,betas=(0.5,0.999))


import torchvision.utils as vutils

for epoch in range(50):
    for i , (data,labels) in enumerate(dataloader,0):
        D.zero_grad()
        
        real_images = data
        Labels= labels
        Input = Variable(real_images)
        Input_size=Input.size()[0]
        
        c_one_hot = torch.from_numpy(label_binarizer.transform(Labels))
        c_emb_data = c_one_hot.type(torch.cuda.FloatTensor)       
        
        target = Variable(torch.ones(Input_size)).cuda()#,dtype=torch.long)).cuda(async = True)
        Input=Input.cuda()
        output = D(Input.float(),c_emb_data)

        errD_real = criterion(output,target)#criterion(output,target)
#        output[output==0]= 1
#        edr = -torch.log(output+eps)
#        edr[torch.isinf(edr)] = 1
        
        
        
        nz = Variable(torch.randn(Input_size,100,1,1)).cuda()
         
        fake_images = G(nz.float(),c_emb_data.view((len(data),11,1,1)))
        
        target_f = Variable(torch.zeros(Input_size)).cuda()#Variable(torch.zeros(Input_size))
        
        output_f = D(fake_images.detach().float(),c_emb_data)
        
        errD_fake =  criterion(output_f,target_f)
#        eDf = -torch.log( 1-output_f)
#        eDf[torch.isinf(eDf)] = 1
        
        

        errD =errD_real+errD_fake
#        errD = torch.mean(edr+eDf+edft)
        #print(NLL_Loss_D(errD_real.detach().cpu(),errD_fake.detach().cpu(),errD_fake_true.detach().cpu()))
        
        #tt=NLL_Loss_D(output.detach().cpu(),output_f.detach().cpu(),output_ft.detach().cpu())
        
        errD.backward()
        Disc_optimazer.step()
        
        G.zero_grad()
        
        fake = G(nz.float(),c_emb_data.view((len(data),11,1,1)))
        target_G =Variable(torch.ones(Input_size)).cuda() #Variable(torch.ones(Input_size))
        output_G = D(fake.float(),c_emb_data)
#        print(output_G)
        errG=criterion(output_G,target_G)#NLL_Loss_G(output_G.detach().cpu().data.numpy())#
        #Glog=torch.log(output_G+eps)
        #Glog[torch.isinf(Glog) | torch.isnan(Glog)]=1
        #errG  = -torch.mean(Glog)
        #print(errG)
        errG.backward() 
        Gen__optimazer.step()
        
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' %(epoch,50,i,len(dataloader),errD.data,errG.data))
        
        if i % 32 == 0:
            vutils.save_image(real_images,'%s/real_samples.png' % "./results/class",normalize = True)
            
            vutils.save_image(fake.data,'./results/class/false_'+str(epoch)+'_'+str(i)+'_samples.png',normalize = True)
    
    if epoch % 2 == 0 or (epoch+1) % 50 == 0:
        torch.save(G.state_dict(), './results/class/models/'+str(epoch+1)+'.pth')
        torch.save(D.state_dict(), './results/class/models/'+str(epoch+1)+'.pth') 

