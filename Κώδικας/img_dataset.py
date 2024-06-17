# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 01:25:55 2021

@author: nikolaos damianos

Auto to arxeio einai gia na katevasei tis eikones apo to imagenet
kai na kanei augmentation
"""

URL = "http://imagenet.stanford.edu/api/text/imagenet.synset.geturls?wnid="


diri ='./data/' #path dataset

#classes_info_txt = './data/MindBigData-Imagenet/WordReport-v1.041.txt'
classes_info_txt = './dataset40Classes.txt'

file1 = open(classes_info_txt, 'r') 
Lines = file1.readlines() 
  
file2 = open('./finished.txt', 'r') 
Lines2 = file2.readlines() 


_classes_ = {}

class_names = []



count = 0
classes = []
#num_of_eeg_files_per_class = []
finished_classes = []

for line in Lines: 
    print("Line{}: {}".format(count, line.strip())) 
    #class  | num of eeg files | class_code 
    line__ = line.strip().split(',')
    classes.append(line__[1])
    count+=1

cnt=0 
for line in Lines2: 
    print("Line{}: {}".format(cnt, line.strip())) 
    #class  | num of eeg files | class_code 
    line__ = line.strip()
    finished_classes.append(line__)
    cnt+=1


    

import os
from PIL import Image
folders = []

        
for name in classes:
    folder_name ='./data/Images/'+name
    folders.append(folder_name)
    try:
        os.mkdir(folder_name)
    except OSError:
        print ("The folder failed to be created")
    else:
        print ("Successfully created the folder "+  folder_name)
        


def folder_find(class_name):
    for f in folders:
        if class_name in f:
            return f
    return diri+'Images/'


import random
import os
from io import BytesIO
import requests
import random
from urllib import request
from bs4 import BeautifulSoup
import numpy as np

def imagenet_download(link,name,img_name):

        
    try:
        response = request.urlopen(link)
        image_data = response.read()
    except:
        print('Warning: Could not download image from {}'.format(link))
        return -1
    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse this image')
        return -1
                          
        
    try:
        pil_image = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image to RGB')
        return -1
       
    try:
        pil_image = pil_image.resize((64, 64))
    except:
        print('Warning: Failed to resize image to 64x64')
        return -1
    
    try:
        pil_image.save(folder_find(name)+'/'+img_name+'.jpg', format='JPEG', quality=90)
    except:
        print('Warning: Failed to save image {}'.format(folder_find(name)+'/'+img_name+'.jpg'))    
        return -1
    
    return 0
                
       
#def loader():
#    
#    for name in classes:
#        page = requests.get(URL+name)
#        soup= BeautifulSoup(page.content,"lxml")
#        links =str(soup.find_all('p')[0]).splitlines()
#        links[0] = links[0][3:]
#        del links[-1]
#        
#        images_urls = random.choices(links, k=300)
#        
#        img_num = 1
#        for link in images_urls:
#            
#            if(imagenet_download(link,name,str(img_num)) != -1):
#                print(name + ' : ' + str(img_num) +  'out of 300!')
#                img_num += 1
#            else :
#                print('failed to download')




import glob
def find_nums(Class):
     data_dir = 'C:\\Users\\nikolaos damianos\\Desktop\\eeg\\data\\Images\\'+Class+'\\*'
     images = glob.glob(data_dir)
     
     return len(images)

def Downloader():
    
    
    for name in classes:
        if name in finished_classes:
            continue
        num = find_nums(name)
        page = requests.get(URL+name)
        soup= BeautifulSoup(page.content,"lxml")
        links =str(soup.find_all('p')[0]).splitlines()
        links[0] = links[0][3:]
        del links[-1]
        
        images_urls = links#random.choices(links, k=500)
        
        if name == 'n02124075':
            del images_urls[:330]
        
        
        if name == 'n04086273':
            del images_urls[:590]
        
        if name == 'n03063599':
            del images_urls[:775]
        if name in ['n03297495']:
            del images_urls[195]
            del images_urls[196]
        
        
        img_num = num
        i = num
#        used_links = []
        
        for i in range(len(images_urls)): 
            if(imagenet_download(images_urls[i],name,str(img_num)) != -1):
                print(name + ' : ' + str(img_num) +  'out of '+ str(len(images_urls))+' !')
                # used_links.append(link)
                img_num += 1
                i+=1
            else :
                print('failed to download')
        
        finished_classes.append(name)
        np.savetxt('finished.txt', finished_classes, delimiter=',', fmt="%s") 
            


#Downloader() 
from PIL import Image,ImageOps

def img_augment():
    diri2 ='.\\data\\64_64_2\\*'
    img_classes=glob.glob(diri2)
    
    for Class in img_classes:

        images = [i for i in range(1,51)]
        
        for img in images:
            im = Image.open(Class+'\\'+str(img)+'.jpg')
#            im.save(Class+'\\'+img, "JPEG")
            im_mirror = np.array(ImageOps.mirror(im).resize((96,96), resample=Image.ANTIALIAS))
            im_rescaled = np.array(im.resize((96,96), resample=Image.ANTIALIAS))
            random_seg = np.random.randint(96-64,size=3)
            
            pp1 = Image.fromarray(im_rescaled[random_seg[0]:64+random_seg[0],random_seg[0]:64+random_seg[0],:])
            pp1.save(Class+'\\'+str(img+50)+'.jpg', "JPEG")
            
            pp2 = Image.fromarray(im_rescaled[random_seg[1]:64+random_seg[1],random_seg[1]:64+random_seg[1],:])
            pp2.save(Class+'\\'+str(img+100)+'.jpg', "JPEG")
            
            pp3 = Image.fromarray(im_rescaled[random_seg[2]:64+random_seg[2],random_seg[2]:64+random_seg[2],:])
            pp3.save(Class+'\\'+str(img+150)+'.jpg', "JPEG")
            
            random_seg_m = np.random.randint(96-64,size=2)
            mpp1= Image.fromarray(im_mirror[random_seg_m[0]:64+random_seg_m[0],random_seg_m[0]:64+random_seg_m[0],:])
            mpp1.save(Class+'\\'+str(img+200)+'.jpg', "JPEG")
            
            mpp2= Image.fromarray(im_mirror[random_seg_m[1]:64+random_seg_m[1],random_seg_m[1]:64+random_seg_m[1],:])
            mpp2.save(Class+'\\'+str(img+250)+'.jpg', "JPEG")
            


def img_augment2():
    diri2 ='.\\data\\Images\\*'
    img_classes=glob.glob(diri2)
    
    max_len = 0
    
    for Class in img_classes:
        if max_len < len(glob.glob(Class+'\\*')):
            max_len = len(glob.glob(Class+'\\*'))
            
    for Class in img_classes:

        images = glob.glob(Class+'\\*') #[i for i in range(1,51)]
        img_names =  [int(i.split('\\')[-1].split('.')[0]) for i in images]
        
        if len(images) < max_len:
            random_ = np.random.randint(len(images) , size=max_len-len(images))
            
            imgs = [img_names[im] for im in random_]
            cnt=len(images)
            for img in imgs:
                im = Image.open(Class+'\\'+str(img)+'.jpg')
                im_rescaled = np.array(im.resize((96,96), resample=Image.ANTIALIAS))
                random_seg = np.random.randint(96-64)
                pp1 = Image.fromarray(im_rescaled[random_seg:64+random_seg,random_seg:64+random_seg,:])
                pp1.save(Class+'\\'+str(cnt)+'.jpg', "JPEG")
                cnt+=1
            
        



img_augment()
img_augment2()




    
    
    
    
    