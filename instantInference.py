
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import soccerSegment
from utilities import getDev, setSeed, plot_learning_curve
import os
from utilities import saveImagesInference,saveDetectedInference,saveSegmentedInference,plot_det_seg_visualsInference
import matplotlib.pyplot as plt

avDev = getDev()
print(avDev)
avDev = "cpu"
seed = 1
setSeed(seed)
lr =0.01

import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
model = soccerSegment(resnet18, [5, 6, 7, 8], [64, 128, 256, 256, 0], [512, 256, 256, 128], [512, 512, 256], 256)
model.to(avDev)

checkpoint_path = './checkpoints/checkpoint'
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        seglosses = checkpoint['seglosses']
        detlosses = checkpoint['detlosses']
        print("Checkpoint Loaded")
transform_input = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                    std=[0.5, 0.5, 0.5])])   
folder_with_images = './sample_images'

inputs=[]
det =[]
seg =[]
for r, _, f in os.walk(folder_with_images):
    for file in f:
        image = Image.open(os.path.join(r, file))
        transformed_image = transform_input(image)
        segmented,detected = model(torch.unsqueeze(transformed_image,0))
        inputs.append(saveImagesInference(transformed_image))
        det.append(saveDetectedInference(detected[0]))
        seg.append(saveSegmentedInference(segmented[0]))
plot_det_seg_visualsInference(inputs,det,seg,folder_with_images,"outputs.png")
    
        
            
            
        
