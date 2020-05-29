import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

path='D:/Datascience/Sign-Language-Alphabet-recognizer/mydata/'
path2='D:/Datascience/Sign-Language-Alphabet-recognizer/preprocessed'

gestures=os.listdir(path)

print(gestures)

for ix in gestures:
    images=os.listdir(path+ix)
    os.mkdir(path2+ix)
    for cx in images:
        img_path=path+ix+'/'+cx
        img=cv2.imread(img_path)
        grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thresh=cv2.threshold(grey,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        save_img=cv2.resize(thresh,(50,50))
        cv2.imwrite(path2+ix+'/'+cx,save_img)

