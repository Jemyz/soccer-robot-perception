#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[126]:

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import random
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from hyperopt import fmin, tpe, hp
import time
import copy
import os
import webcolors
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

# In[129]:


def showImagesWithTargets(img,targets):
    img = img / 2 + 0.5     # unnormalize
    img = img.numpy()
    plt.imshow(img.transpose((1,2,0)))
    plt.show()
    targets=targets.numpy()
    plt.imshow(targets.transpose((1,2,0)))
    plt.show()


# In[149]:


def showImagesWithTargets_Segmentation(img,targets):
    img = img / 2 + 0.5     # unnormalize
    img = img.numpy()
    plt.imshow(img.transpose((1,2,0)))
    plt.show()
    targets=targets.numpy()
    plt.show()


# In[152]:


#Visualise segmented output
def visualiseSegmented(segmentedImage,iter):
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
    plt.savefig('./outputs/segmented/img[' + str(iter) + '].png')
    plt.show()

def visualiseSegmentedTarget(target,iter):
    colorMap = [[0,0,0],[255,255,255],[0,255,0]]
    size = target.size()
    segImage = np.empty([3,size[0],size[1]])
    values = torch.unique(target)
    for i in range(size[0]):
        for j in range(size[1]):
            label = np.argwhere(values == target[i][j]) 
            if label >= 3:
                label =2
            color = colorMap(label)
            for k in range(3):
                segImage[k,i,j] = color[k]
    segImage = segImage.transpose((1,2,0))
    plt.imshow(segImage)
    plt.show()

# In[131]:


def plot_error_curve(errors):
  plt.suptitle('Learning Curve', fontsize=20)
  plt.xlabel('Iterations', fontsize=18)
  plt.ylabel('Classification Error', fontsize=16)
  plt.plot(np.array(errors))


# # Load Data Functions

# In[132]:


def transformations(listTransforms):
    return transforms.Compose(listTransforms)


# In[133]:


def get_colour_name(rgb_triplet):
    min_colours = {}
    for key, name in webcolors.CSS21_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb_triplet[0]) ** 2
        gd = (g_c - rgb_triplet[1]) ** 2
        bd = (b_c - rgb_triplet[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


# In[134]:


def get_labels(images):
    size = images.shape
    targets = np.empty([size[0],size[2],size[3]]).astype(int)
    targets.fill(2)
    for i in range(size[0]):
        img = images[i].numpy().transpose((1,2,0))
        indices = np.where((img[:,:,0]==0) & (img[:,:,1]==0) & (img[:,:,2]==0))
        coordinates = [indices[0],indices[1]]
        targets[i,coordinates] = 0
        indices = np.where((img[:,:,0]==128) & (img[:,:,1]==128) & (img[:,:,2]==0))
        coordinates = [indices[0],indices[1]]
        targets[i,coordinates] = 1
        indices = np.where((img[:,:,0]==0) & (img[:,:,1]==128) & (img[:,:,2]==0))
        coordinates = [indices[0],indices[1]]
        targets[i,coordinates] = 2
    return targets
        


# In[135]:


class CudaVisionDataset(Dataset):

    def __init__(self, dir_path,task="detection",transform=None):
        super(CudaVisionDataset, self).__init__()
        self.img_paths,self.target_paths = read_files(img_dir_path=dir_path)
        self.transform = transform
        self.task = task
          
    def __getitem__(self, index):
        
        input_img = Image.open(self.img_paths[index])
        #print(input_img.size)
        input_img = transforms.functional.resize(input_img,(640,480))

        target_img = Image.open(self.target_paths[index])
        target_img = transforms.functional.resize(target_img,(160,120))
        if self.transform != None:
            
            trnfm_input = transformations(self.transform)
            input_img = trnfm_input(input_img)
            target_transform  = copy.copy(self.transform)
            if self.task == "segmentation":
                target_transform.pop()
                trnfm_target= transformations(target_transform)
                target_img1  = trnfm_target(target_img)
                target_img_temp = torch.squeeze(target_img1)
                target_img = torch.ones([160,120], dtype=torch.float64)
                values = torch.unique(target_img_temp)
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
                target_transform = [transforms.Normalize(mean=[0.5],
                                         std=[0.5])]
                
            else:    
                trnfm_target= transformations(target_transform)
                target_img  = trnfm_target(target_img)
               
            #print(target_img.shape)
        return input_img,target_img

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
        for file in f:
            img_paths.append(os.path.join(r, file))
            
    if len(img_paths) == 0:
        print("No Images in given path available. Check directory or format.")
    dir = os.path.join(img_dir_path,'output')
    for r, _, f in os.walk(dir):
        for file in f:
            target_paths.append(os.path.join(r, file))
    if len(target_paths) == 0:
        print("No Images in given path available. Check directory or format.")
    
    return img_paths,target_paths





lr = 0.001
batchSize = 20
epoch = 50
tv_weight = 0.2


# # Load Data And Make Iterable

# In[138]:


listTransforms_train = [transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]
listTransforms_test = [transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]

# In[139]:


class CudaVisionDataLoader:

    def __call__(self, dir_path='./small_data', task="detection",transform=None,batch_size=20):
        dataset = CudaVisionDataset(dir_path,task,transform)
        return DataLoader(dataset,batch_size=batch_size, shuffle=True)


# In[140]:


#listTransforms = [transforms.RandomHorizontalFlip(0.5),
 #       transforms.ToTensor(),
  #      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
#transforms(listTransforms)
data = CudaVisionDataLoader()
parentDir = './small_data'
dirDetectionDataset = os.path.join(parentDir,'detection')
dirSegmentationDataset = os.path.join(parentDir,'segmentation')
train_loader_detection = data.__call__(os.path.join(dirDetectionDataset,'train'),"detection",listTransforms_train,batchSize)
test_loader_detection = data.__call__(os.path.join(dirDetectionDataset,'test'),"detection",listTransforms_test,batchSize)
train_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset,'train'),"segmentation",listTransforms_train,batchSize)
test_loader_segmentation = data.__call__(os.path.join(dirSegmentationDataset,'test'),"segmentation",listTransforms_test,batchSize)


# In[141]:


dataiter_detection = train_loader_detection.__iter__()

dataiter_segmentation = train_loader_segmentation.__iter__()
images, targets = dataiter_segmentation.next()


# # Model Definition

# In[142]:


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

        self.conv_det = nn.Conv2d(last_layer_arch, 3, kernel_size=1, stride=1, padding=0)
        self.conv_seg = nn.Conv2d(last_layer_arch, 3, kernel_size=1, stride=1, padding=0)

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
        seg = nn.functional.softmax(seg)
        det = self.conv_det(skip_links[i])
        return seg, det


# In[ ]:





# # Training Cycle

# In[143]:


#Metrics 

def segmentationAccuracy(segmented,targets):
    sizes = segmented.size()
    
    total_pixel = sizes[0] * sizes[1] *sizes[2]
    difference = torch.abs(segmented-targets)
    same =(difference==0).sum()
    
    accuracy = (same.item())/total_pixel *100
    return accuracy
    


# In[144]:


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


# In[155]:


#Training cycle without hyperopt
def train():
    import torchvision.models as models
    resnet18  = models.resnet18(pretrained=True)
    model = soccerSegment(resnet18,[5,6,7,8],[64, 128, 256, 256, 0],[512, 256, 256, 128],[512, 512, 256],256)
    #if torch.cuda.is_available():
    #    model.cuda()
    iter = 0
    criterionDetection = nn.MSELoss()
    criterionSegmented = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    
    for num in range(epoch):
        
        try:
            images,targets = dataiter_segmentation.next()
            if torch.cuda.is_available():
                images = images.to(avDev)
                targets = targets.to(avDev)
        except:
            dataiter_segmentation = train_loader_segmentation.__iter__()
            images,targets = dataiter_segmentation.next()
            if torch.cuda.is_available():
                images = images.to(avDev)
                targets = targets.to(avDev)
          
          # Forward pass to get output
        segmented,detected = model(images)
        
        targets_m= torch.squeeze(targets)
        visualiseSegmented(targets_m[1],iter)
        loss = criterionSegmented(segmented, targets_m.long())
        
          # Getting gradients w.r.t. parameters
        loss.backward()

          # Updating parameters
        optimizer.step()
        iter += 1
        losses =0
        if iter % 1 == 0:
              # Iterate through test dataset
          
          
            for images, targets in test_loader_segmentation:
                  #######################
                  #  USE GPU FOR MODEL  #
                  #######################
                model.eval()
                with torch.no_grad():
                    if torch.cuda.is_available():
                        images = images.to(avDev)
                        targets = targets.to(avDev)

                  # Forward pass only to get logits/output
                
                    segmented,detected = model(images)


                  # Total number of labels
                    targets_m= torch.squeeze(targets)
                    loss=criterionSegmented(segmented, targets_m.long())
                    losses = losses+loss.item()    
            
              # Print Loss
                    print('Loss Segmentation: {}.', loss.item())
                    segmentedLabels=torch.argmax(segmented,dim=1)
                    accuracy =segmentationAccuracy(segmentedLabels.long(),targets_m.long())
                    print('Segmentation Accuracy: {}.',accuracy)
            #Save some images to check
                    #visualiseSegmented(targets_m[0],iter)
                    visualiseSegmented(segmentedLabels[1],iter)
            


# In[156]:


train()


