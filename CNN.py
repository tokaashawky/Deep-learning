import pandas as pd 
import os
import cv2



dirr=''
categories= ['benign','malignant']
train_data=[]
img_size=300

for category in categories:
    path=os.path.join(dirr,category)
    claass=categories.index(category)
    for image in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,image))
        new_array=cv2.resize(img_array,(img_size,img_size))
        train_data.append([new_array,claass])
        

































