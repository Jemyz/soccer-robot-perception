from PIL import Image
import cv2
import os,fnmatch,shutil
import numpy as np
# parse an xml file by name
directory = "C:\\Users\Sarah Khan\Downloads\\bigcuda5.informatik.uni-bonn.de+8686\segmentation\dataset\image"
for file in os.listdir(directory):
    if fnmatch.fnmatch(file,"*.jpg"):
        shutil.copy(os.path.join(directory,file),os.path.join("C:\\Users\Sarah Khan\Downloads\data\segment\input",file))
    if fnmatch.fnmatch(file,"*.png"):

        shutil.copy(os.path.join(directory,file), os.path.join("C:\\Users\Sarah Khan\Downloads\data\input", file))

directory = "C:\\Users\Sarah Khan\Downloads\\bigcuda5.informatik.uni-bonn.de+8686\segmentation\dataset\\target"
for file in os.listdir(directory):
    if fnmatch.fnmatch(file,"*.png"):
        image = cv2.imread(
            os.path.join(directory,file))
        dimensions = image.shape
        image = cv2.resize(image,(int(dimensions[1]*25/100),int(dimensions[0]*25/100)), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join("C:\\Users\Sarah Khan\Downloads\\data\segment\output", file),image)