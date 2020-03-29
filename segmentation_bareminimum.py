#!/usr/bin/env python
# coding: utf-8

# # Import Modules


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from hyperopt import fmin, tpe, hp
import time
import copy
import os
from PIL import Image
from torchvision.utils import save_image


# # Check Available Devices
# 

# In[127]:

torch.cuda.empty_cache()
if torch.cuda.is_available:
  avDev = torch.device("cuda")
else:
  avDev = torch.device("cpu")
print(avDev)
avDev = "cpu"


# # Set Seed

# In[128]:


seed = 1
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True


# # Utility for Displaying Images, Error Curve



# In[149]:


def showImages(img,iter,fileName):
    img = img / 2 + 0.5     # unnormalize
    img = img.numpy()
    img = img.transpose((1,2,0))
    plt.imshow(img)
    plt.savefig('./outputs/segmented/img_input '+fileName+'[' + str(iter) + '].png')
    plt.show()


# In[152]:


#Visualise segmented output
def visualiseSegmented(segmentedImage,iter,stringName,fileName):
    colorMap = [[0,0,0],[255,255,255],[0,255,0]]
    size = segmentedImage.size()
    segImage = np.empty([3,size[0],size[1]])
    for i in range(size[0]):
        for j in range(size[1]):
            color = colorMap[int(segmentedImage[i][j])]
            for k in range(3):
                segImage[k,i,j] = color[k]
    segImage = segImage.transpose((1,2,0))
    plt.imshow(segImage)
    plt.savefig('./outputs/segmented/img_'+ stringName+"_"+ fileName+'[' + str(iter) + '].png')
    plt.show()




# # Load Data Functions

# In[132]:


def transformations(listTransforms):
    return transforms.Compose(listTransforms)


class CudaVisionDataset(Dataset):

    def __init__(self, dir_path,task="detection",transform=None):
        super(CudaVisionDataset, self).__init__()
        self.img_paths,self.target_paths = read_files(img_dir_path=dir_path)
        self.transform = transform
        self.task = task
          
    def __getitem__(self, index):
        
        input_img = Image.open(self.img_paths[index])
        iname = os.path.basename(self.img_paths[index])
        #print(input_img.size)
        #showImages(input_img)
        input_img = transforms.functional.resize(input_img,(480,640))
        #showImages(input_img)
        target_img = Image.open(self.target_paths[index])
        oname = os.path.basename(self.target_paths[index])
        target_img = transforms.functional.resize(target_img,(120,160))
        if self.transform != None:
            
            trnfm_input = transformations(self.transform)
            input_img = trnfm_input(input_img)
            target_transform  = copy.copy(self.transform)
            if self.task == "segmentation":
                target_transform.pop()
                trnfm_target= transformations(target_transform)
                target_img1  = trnfm_target(target_img)
                target_img_temp = torch.squeeze(target_img1)
                target_img = torch.ones([120,160], dtype=torch.float64)
                values= torch.tensor([0.0000,1.0,2.0,3.0])
                target_img_temp = target_img_temp *255
                index = (target_img_temp == values[0]).nonzero()
                target_img[index[:,0],index[:,1]] =0
                index = (target_img_temp == values[1]).nonzero()
                target_img[index[:,0],index[:,1]] =2
                index = (target_img_temp == values[2]).nonzero()
                target_img[index[:,0],index[:,1]] =1
                index = (target_img_temp == values[3]).nonzero()
                target_img[index[:,0],index[:,1]] =2
            else:    
                trnfm_target= transformations(target_transform)
                target_img  = trnfm_target(target_img)
               
            #print(target_img.shape)
        return input_img,target_img,iname,oname

    def __len__(self):
        return len(self.target_paths)


# In[136]:


def read_files(img_dir_path):
    img_paths = []
    target_paths = []
    if os.path.isdir(img_dir_path):
        print("Folder exists. Reading..")
    dir = os.path.join(img_dir_path,'input')
    for r, _, f in os.walk(dir):
        f.sort()
        for file in f:
            img_paths.append(os.path.join(r, file))
            
    if len(img_paths) == 0:
        print("No Images in given path available. Check directory or format.")
    dir = os.path.join(img_dir_path,'output')
    for r, _, f in os.walk(dir):
        f.sort()
        for file in f:
            target_paths.append(os.path.join(r, file))
    if len(target_paths) == 0:
        print("No Images in given path available. Check directory or format.")
    
    return img_paths,target_paths





lr = 0.001
batchSize = 20
epoch = 100



# # Load Data And Make Iterable

# In[138]:


listTransforms_train = [transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])]
listTransforms_test = [transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])]

# In[139]:


class CudaVisionDataLoader:

    def __call__(self, dir_path='./small_data', task="detection",transform=None,batch_size=20):
        dataset = CudaVisionDataset(dir_path,task,transform)
        return DataLoader(dataset,batch_size=batch_size, shuffle=True)


# In[140]:

data = CudaVisionDataLoader()
parentDir = './small_data'
dirSegmentationDataset = os.path.join(parentDir,'segmentation')
train_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset,'train'),"segmentation",listTransforms_train,batchSize)
test_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset,'test'),"segmentation",listTransforms_test,batchSize)
dataiter_segmentation = train_loader_segmentation.__iter__()


class ResNet18(nn.Module):
    def __init__(self, original_model, outputs_indices):
        super(ResNet18, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        self.outputs_indices = [0] + outputs_indices
        print(self.outputs_indices)

    def forward(self, x):
        out = []

        for i in range(len(self.outputs_indices) - 1):
            x = self.features[self.outputs_indices[i]:self.outputs_indices[i + 1]](x)
            out.append(x)
        return out

                
class LocationAwareConv2d(torch.nn.Conv2d):
    def __init__(self,gradient,w,h,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.locationBias=torch.nn.Parameter(torch.zeros(w,h,3))
        self.locationEncode=torch.autograd.Variable(torch.ones(w,h,3))
        if gradient:
            for i in range(w):
                self.locationEncode[i,:,1]=(i/float(w-1))
            for i in range(h):
                self.locationEncode[:,i,0] = (i/float(h-1))
    def forward(self,inputs):
        if self.locationBias.device != inputs.device:
            self.locationBias=self.locationBias.to(inputs.get_device())
        if self.locationEncode.device != inputs.device:
            self.locationEncode=self.locationEncode.to(inputs.get_device())
        b=self.locationBias*self.locationEncode
        return super().forward(inputs)+b[:,:,0]+b[:,:,1]+b[:,:,2]

class soccerSegment(nn.ModuleList):
    def __init__(self, resnet18, outputs_indices, skips_arch, deconvs_arch, bn_arch, last_layer_arch):
        super(soccerSegment, self).__init__()
        self.resnet18 = ResNet18(resnet18, outputs_indices)
        
        # skips_arch = [64, 128, 256, 256, 0]
        # deconvs_arch = [512, 256, 256, 128]
        # bn_arch = [512, 512, 256]
        # last_layer_arch = 256
        self.skips = nn.ModuleList(
            [nn.Conv2d(skips_arch[i], skips_arch[i + 1], kernel_size=1, stride=1, padding=0) for i in
             range(len(skips_arch) - 2)])

        self.deconvs = nn.ModuleList(
            reversed([nn.ConvTranspose2d(deconvs_arch[i] + skips_arch[len(skips_arch) - i - 1], deconvs_arch[i + 1],
                                         kernel_size=2, stride=2, padding=0) for i in
                      range(len(deconvs_arch) - 1)]))

        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=bn_arch[i]) for i in
             reversed(range(len(bn_arch)))])
        self.relu = nn.ReLU()

        self.conv_det = LocationAwareConv2d(True,120,160,last_layer_arch, 3,kernel_size=1, stride=1, padding=0)
        self.conv_seg = LocationAwareConv2d(True,120,160,last_layer_arch,3, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        skip_links = self.resnet18(x)

        for i in reversed(range(len(skip_links))):
            if i == len(skip_links) - 1:
                skip_links[i - 1] = torch.cat(
                    (self.skips[i - 1](skip_links[i - 1]), self.deconvs[i - 1](self.relu(skip_links[i]))),
                    1)
            elif i == 0:
                skip_links[i] = self.relu(self.bns[i](skip_links[i]))
            else:
                skip_links[i - 1] = torch.cat(
                    (self.skips[i - 1](skip_links[i - 1]),
                     self.deconvs[i - 1](self.relu(self.bns[i](skip_links[i])))),
                    1)
        seg = self.conv_seg(skip_links[i])
        det = self.conv_det(skip_links[i])
        return seg, det
"""
class LocationAwareConv2d(torch.nn.Conv2d):
    def __init__(self,gradient,w,h,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.locationBias=torch.nn.Parameter(torch.zeros(w,h,3))
        self.locationEncode=torch.autograd.Variable(torch.ones(w,h,3))
        if gradient:
            for i in range(w):
                self.locationEncode[i,:,1]=(i/float(w-1))
            for i in range(h):
                self.locationEncode[:,i,0] = (i/float(h-1))
    def forward(self,inputs):
        if self.locationBias.device != inputs.device:
            self.locationBias=self.locationBias.to(inputs.get_device())
        if self.locationEncode.device != inputs.device:
            self.locationEncode=self.locationEncode.to(inputs.get_device())
        b=self.locationBias*self.locationEncode
        return super().forward(inputs)+b[:,:,0]+b[:,:,1]+b[:,:,2]

class soccerSegment(nn.ModuleList):
    def __init__(self, resnet18):
        super(soccerSegment, self).__init__()
        self.resnet18 =resnet18
        #self.layer4 = nn.Sequential(*list(resnet18.children())[:-2])
        self.preconv = nn.Conv2d(3,64,kernel_size=7,stride =2,padding=3)
        self.prebatch = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=1,stride=2,padding=0)
        self.relu = nn.ReLU()
        self.deconv1_2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2,padding=0)
        #self.deconv2_3 =  nn.ConvTranspose2d(256,256,kernel_size=2,stride=2,padding=0)
        self.deconv3 = nn.ConvTranspose2d(512,128,kernel_size=2,stride=2,padding=0)
        self.batch1_2 = nn.BatchNorm2d(512)
        self.batch3 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(64,128,kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(128,256,kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256,256,kernel_size=1, stride=1, padding=0)
        self.conv_det = LocationAwareConv2d(True,120,160,256, 3,kernel_size=1, stride=1, padding=0)
        self.conv_seg = LocationAwareConv2d(True,120,160,256,3, kernel_size=1, stride=1, padding=0)
        
    def forward(self,x):
        print(x.shape)
        l1 = self.resnet18.layer1(self.pool(self.prebatch(self.preconv(x))))
        print(l1.shape)
        l2 = self.resnet18.layer2(l1)
        print(l2.shape)
        l3 = self.resnet18.layer3(l2)
        l4 = self.relu(self.resnet18.layer4(l3))
        #l2=self.conv2(l2)
        d1 = self.deconv1_2(l4)
        con1 = self.relu(self.batch1_2(torch.cat((d1,self.conv3(l3)),1)))
        d2 = self.deconv1_2(con1)
        print(d2.shape)
        con2 = self.relu(self.batch1_2(torch.cat((d2,self.conv2(l2)),1)))
        d3 = self.deconv3(con2)
        con3 = self.relu(self.batch3(torch.cat((d3,self.conv1(l1)),1)))
        seg = self.conv_seg(con3)
        det = self.conv_det(con3)
        return seg, det
        
  """      
        

# In[ ]:



#Metrics 

def segmentationAccuracy(segmented,targets):
    sizes = segmented.size()
    
    total_pixel = sizes[0] * sizes[1] *sizes[2]
    difference = torch.abs(segmented-targets)
    same =(difference==0).sum()
    
    accuracy = (same.item())/total_pixel *100
    return accuracy
    
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=0.0002):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        #Ignored line segmentation in calculations
        h_tv = torch.pow(torch.abs(x[:,0,1:,:]-x[:,0,:h_x-1,:]),2).sum()+torch.pow(torch.abs(x[:,2,1:,:]-x[:,2,:h_x-1,:]),2).sum()
        w_tv = torch.pow(torch.abs(x[:,0,:,1:]-x[:,0,:,:w_x-1]),2).sum()+torch.pow(torch.abs(x[:,2,:,1:]-x[:,2,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*(h_tv/h_x+w_tv/w_x)/batch_size

    def _tensor_size(self,t):
        return (t.size()[1]-1)*t.size()[2]*t.size()[3]



# In[155]:


#Training cycle without hyperopt
def train():
    import torchvision.models as models
    resnet18  = models.resnet18(pretrained=True)
    model = soccerSegment(resnet18,[5,6,7,8],[64, 128, 256, 256, 0],[512, 256, 256, 128],[512, 512, 256],256)
    #model = soccerSegment(resnet18)
    #if torch.cuda.is_available():
     #   model.cuda()
    
    iter = 0
    
    criterionSegmented = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    optimizer.zero_grad()
    for num in range(10):
        flagToRun = 1
        print("This is epoch:",epoch)
        dataiter_segmentation = train_loader_segmentation.__iter__()
        while(flagToRun):
           print("Train Data") 
           try:
               images,targets,input_name,output_name = dataiter_segmentation.next()
               
           except:
               flagToRun=0
           if(flagToRun):
               if torch.cuda.is_available():
                   images = images.to(avDev)
                   targets = targets.to(avDev)
               segmented,detected = model(images)
               segmentedLabels= torch.argmax(segmented,1)
               tvLoss = TVLoss()
               toadd = tvLoss.forward(segmented)
               loss = criterionSegmented(segmented, targets.long()) 
               print("Loss",loss.item())
          # Getting gradients w.r.t. parameters
               loss.backward()

          # Updating parameters
               optimizer.step()
               showImages(images[0],num,input_name[0])
               visualiseSegmented(targets[0],num,"train",output_name[0])
               visualiseSegmented(segmentedLabels[0],num,"output",output_name[0])
           
               
               
              
train()


