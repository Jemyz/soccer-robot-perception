from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import numpy as np
import torch
import copy

#Define gaussian blob size and density
radius = 3
numberOfPoints = 300


class CudaVisionDatasetDetection(Dataset):

    def __init__(self, dir_path,transform=None):
        super(CudaVisionDatasetDetection, self).__init__()
        self.img_paths,self.target_paths = read_files(img_dir_path=dir_path)
        self.transform = transform
     
    def __getitem__(self, index):

        #Get input image
        input_img = Image.open(self.img_paths[index])
        input_img = transforms.functional.resize(input_img,(480,640))
        trnfm_input = transformations(self.transform)
        input_img = trnfm_input(input_img)
        #Get target image from xml
        target_xml = ET.parse(self.target_paths[index])
        root = target_xml.getroot()
        object_name = "dummy"
        list_center = []
        list_class = []
        target_probabilities = np.zeros([3,160,120])
        target_center = np.zeros([3,160,120])
        for elem in root:
            if(elem.tag == "size"):
                for subelem in elem:
                    if(subelem.tag == "width"):
                        width = int(subelem.text)
                    if(subelem.tag == "height"):
                        height = int(subelem.text)
                
            if(elem.tag == "object"):
                for subelem in elem:
                    if(subelem.tag == "name"):
                            object_name = subelem.text
                    if(subelem.tag == "bndbox"):
                        for measurements in subelem:
                            if(measurements.tag == "xmin"):
                                xmin = int(160/width * int(measurements.text))
                            if(measurements.tag == "ymin"):
                                ymin = int(120/height*int(measurements.text))
                            if(measurements.tag == "xmax"):
                                xmax = int(160/width*int(measurements.text))
                            if(measurements.tag == "ymax"):
                                ymax = int(120/height*int(measurements.text))
                        if(object_name == "ball"):
                            center_x = int((xmin+xmax)/2)
                            center_y = int((ymin+ymax)/2)
                            target_probabilities = self.gaussian_blob(target_probabilities,0,center_x,radius,center_y,radius,numberOfPoints)
                            target_center[0][center_x][center_y]=1.0
                        if(object_name == "goalpost"):
                            center_x = int((xmin+xmax)/2)
                            center_y = ymin
                            target_probabilities = self.gaussian_blob(target_probabilities,1,center_x,radius,center_y,radius,numberOfPoints)
                            target_center[1][center_x][center_y]=1.0
                        if(object_name == "robot"):
                            center_x = int((xmin+xmax)/2)
                            center_y = ymin
                            target_probabilities = self.gaussian_blob(target_probabilities,2,center_x,radius+2,center_y,radius+2,numberOfPoints)
                            target_center[2][center_x][center_y]=1.0
        #Interchange height and rows to make compatible with input
        target_probabilities = target_probabilities.transpose((0,2,1))
        target = torch.from_numpy(target_probabilities)
        target_center = target_center.transpose((0,2,1))
        target_center = torch.from_numpy(target_center)
        return input_img,target.float(),target_center


    def gaussian_blob(self,img,classPassed, mean_x, sigma_x, mean_y, sigma_y, number_of_points):
        indicies_x = np.random.normal(mean_x, sigma_x, number_of_points).astype(int)
        indicies_y = np.random.normal(mean_y, sigma_y, number_of_points).astype(int)
        indicies_x[indicies_x < 0] = 0
        indicies_y[indicies_y < 0] = 0
        shape = img.shape
        indicies_x[indicies_x >= shape[1]] = 0
        indicies_y[indicies_y >= shape[2]]= 0
        indicies = [indicies_x, indicies_y]
        img[classPassed][indicies] = 1.0
        return img 


    def __len__(self):
        return len(self.target_paths)
    
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


class CudaVisionDatasetSegmentation(Dataset):
    def __init__(self, dir_path,transform=None):
        super(CudaVisionDatasetSegmentation, self).__init__()
        self.img_paths,self.target_paths = read_files(img_dir_path=dir_path)
        self.transform = transform
          
    def __getitem__(self, index):
        
        input_img = Image.open(self.img_paths[index])
        input_img = transforms.functional.resize(input_img,(480,640))
        target_img = Image.open(self.target_paths[index])
        
        target_img = transforms.functional.resize(target_img,(120,160))
        if self.transform != None:
            
            trnfm_input = transformations(self.transform)
            input_img = trnfm_input(input_img)
            target_transform  = copy.copy(self.transform)
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
            target_img = target_img.long()
        return input_img,target_img
    def __len__(self):
        return len(self.target_paths)

class CudaVisionDataLoader:

    def __call__(self, dir_path='./small_data', task="detection", transform=None, batch_size=20):
        if task == "detection":
            dataset = CudaVisionDatasetDetection(dir_path, transform)
        else:
            dataset = CudaVisionDatasetSegmentation(dir_path, transform)
        return DataLoader(dataset, batch_size, shuffle=True)


def transformations(listTransforms):
    return transforms.Compose(listTransforms)
