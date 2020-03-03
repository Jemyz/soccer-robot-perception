from xml.dom import minidom
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import os,fnmatch,shutil
import numpy as np
# parse an xml file by name
directory = "C:\\Users\Sarah Khan\Downloads\\bigcuda5.informatik.uni-bonn.de+8686\\blob\dataset"
for file in os.listdir(directory):
    if fnmatch.fnmatch(file,"*.jpg"):
        shutil.copy(os.path.join(directory,file),os.path.join("C:\\Users\Sarah Khan\Downloads\data\input",file))
    if fnmatch.fnmatch(file,"*.png"):

        shutil.copy(os.path.join(directory,file), os.path.join("C:\\Users\Sarah Khan\Downloads\data\input", file))
    if fnmatch.fnmatch(file,"*.xml"):

        mydoc = minidom.parse(os.path.join(directory,file))
        width = mydoc.getElementsByTagName('width')
        height = mydoc.getElementsByTagName('height')
        depth = mydoc.getElementsByTagName('depth')
        xmins = mydoc.getElementsByTagName('xmin')
        xmaxs = mydoc.getElementsByTagName('xmax')
        ymins = mydoc.getElementsByTagName('ymin')
        ymaxs = mydoc.getElementsByTagName('ymax')
        names = mydoc.getElementsByTagName('name')
        fileName = mydoc.getElementsByTagName('filename')
        colors = {'R':(0,0,255),'G':(0,255,0),'B':(255,0,0)}

        img = np.zeros([int(width[0].firstChild.data),int(height[0].firstChild.data),int(depth[0].firstChild.data)],dtype=np.uint8)
        img.fill(255)
        window_name = 'Image'
        im = Image.fromarray(img)
        im.save(os.path.join("C:\\Users\Sarah Khan\Downloads\\data\output",fileName[0].firstChild.data)+".jpg")
        i=0
        for name in names:

            image = cv2.imread(os.path.join("C:\\Users\Sarah Khan\Downloads\\data\output", fileName[0].firstChild.data) + ".jpg")
            #cv2.imshow(window_name, image)
            if name.firstChild.data == "ball":
                color = colors['R']
                center_x = int((int(xmins[i].firstChild.data)+int(xmaxs[i].firstChild.data))/2)
                center_y = int((int(ymins[i].firstChild.data)+int(ymaxs[i].firstChild.data))/2)
                image = cv2.circle(image, (center_x,center_y), 5, color, -1)
                #cv2.imshow(window_name,image)
                cv2.imwrite(os.path.join("C:\\Users\Sarah Khan\Downloads\\data\output",fileName[0].firstChild.data)+".jpg", image)
                i=i+1
            if name.firstChild.data == "robot":
                color = colors['B']
                center_x = int((int(xmins[i].firstChild.data) + int(xmaxs[i].firstChild.data)) / 2)
                center_y = int((int(ymins[i].firstChild.data) + int(ymaxs[i].firstChild.data)) / 2)
                image = cv2.circle(image, (center_x, center_y), 5, color, -1)
                cv2.imwrite(os.path.join("C:\\Users\Sarah Khan\Downloads\\data\output", fileName[0].firstChild.data) + ".jpg", image)
                i = i + 1
            if name.firstChild.data == "goalpost":
                color = colors['G']
                center_x = int((int(xmins[i].firstChild.data) + int(xmaxs[i].firstChild.data)) / 2)
                center_y = int((int(ymins[i].firstChild.data) + int(ymaxs[i].firstChild.data)) / 2)
                image = cv2.circle(image, (center_x, center_y), 5, color, -1)
                cv2.imwrite(os.path.join("C:\\Users\Sarah Khan\Downloads\\data\output", fileName[0].firstChild.data) + ".jpg", image)
                i = i + 1
        image = cv2.imread(os.path.join("C:\\Users\Sarah Khan\Downloads\\data\output", fileName[0].firstChild.data) + ".jpg")
        image = cv2.resize(image,(int(int(width[0].firstChild.data)*25/100),int(int(height[0].firstChild.data)*25/100)), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join("C:\\Users\Sarah Khan\Downloads\\data\output", fileName[0].firstChild.data) + ".jpg",
                    image)