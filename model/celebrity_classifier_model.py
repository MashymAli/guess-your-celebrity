from calendar import c
import numpy as np
import cv2 as cv
import matplotlib
from matplotlib import pyplot as plt
#%matplotlib inline


img = cv.imread('model/test/aamir1.jpeg')
#print(img.shape)
#plt.imshow(img)
#plt.show()

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#print(gray.shape)
#print(gray)
#plt.imshow(gray,cmap='gray')
#plt.show()


face_cascade = cv.CascadeClassifier('model/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('model/opencv/haarcascades/haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#print(faces)

(x,y,w,h) = faces[0]  
#print(x,y,w,h)  #x-axis , y-axis , width , height

face_img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#plt.imshow(face_img)
#plt.show()


for (x,y,w,h) in faces:
    face_img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

plt.figure()
#plt.imshow(face_img, cmap='gray') #show whole image
plt.imshow(roi_color, cmap='gray') #show cropped face
plt.show()

