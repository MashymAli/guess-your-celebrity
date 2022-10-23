import numpy as np
import cv2 as cv
import matplotlib
from matplotlib import pyplot as plt
#%matplotlib inline
import os, shutil
import pywt


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

face_img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  #prints rectangle around the face
#plt.imshow(face_img)
#plt.show()


#for (x,y,w,h) in faces:
#    face_img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#    roi_gray = gray[y:y+h, x:x+w]
#    roi_color = face_img[y:y+h, x:x+w]
#    eyes = eye_cascade.detectMultiScale(roi_gray)
#    for (ex,ey,ew,eh) in eyes:
#        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        

#plt.figure()
#plt.imshow(face_img, cmap='gray') #show whole image
#plt.imshow(roi_color, cmap='gray') #show cropped face
#plt.show()

def get_cropped_image_if_2_eyes(image_path):
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

#original image with two eyes
#original_image = cv.imread('model/test/aamir1.jpeg')
#plt.imshow(original_image)
#plt.show()

#cropped image with two eyes
#cropped_image = get_cropped_image_if_2_eyes('./model/test/aamir1.jpeg')
#plt.imshow(cropped_image)
#plt.show()

#original image without two eyes
#original_image_obstructed = cv.imread('model/test/aamir3.jpg')
#plt.imshow(original_image_obstructed)
#plt.show()

#cropped image without two eyes
#cropped_image_no_2_eyes = get_cropped_image_if_2_eyes('model/test/aamir3.jpg')
#print(cropped_image_no_2_eyes) #prints None because this picture has no 2 eyes


#store cropped images in cropped folder inside dataset folder
path_to_data = "./model/dataset/"
path_to_cr_data = "./model/dataset/cropped/"


img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)
print(img_dirs)  #print array/list of all folder names in dataset folder


#creating and removing cropped folder
if os.path.exists(path_to_cr_data):
     shutil.rmtree(path_to_cr_data) #if cropped folder exists remove it, helps when doing multiple runs
os.mkdir(path_to_cr_data) # make cropped folder


cropped_image_dirs = []
celebrity_file_names_dict = {}

for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]
    #print(celebrity_name)
    
    celebrity_file_names_dict[celebrity_name] = []
    
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)  #entry.path means path of that image
        if roi_color is not None:
            cropped_folder = path_to_cr_data + celebrity_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print("Generating cropped images in folder: ",cropped_folder)
                
            cropped_file_name = celebrity_name + str(count) + ".png"
            cropped_file_path = cropped_folder + "/" + cropped_file_name 
            
            cv.imwrite(cropped_file_path, roi_color)
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
            count += 1  


# Preprocessing: Use wavelet transform as a feature for traning our model
def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv.cvtColor( imArray,cv.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)

    return imArray_H


#im_har = w2d(cropped_image,'db1',5)
#plt.imshow(im_har, cmap='gray')
#plt.show()


print(celebrity_file_names_dict)

class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1
print(class_dict)


X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv.imread(training_image)
        if img is None:
            continue
        scalled_raw_img = cv.resize(img, (32, 32))
        img_har = w2d(img,'db1',5)
        scalled_img_har = cv.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
        X.append(combined_img)
        y.append(class_dict[celebrity_name]) 


print(len(X))
X = np.array(X).reshape(len(X),4096).astype(float)
print(X.shape)


# DATA CLEANING IS DONE AT THIS POINT

# TRAIN MODEL STARTING

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
print(len(X_test))

print("\n")
print(classification_report(y_test, pipe.predict(X_test)))

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto',probability=True),
        'params' : {
            'svc__C': [1,10,100,1000],
            'svc__kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'randomforestclassifier__n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'logisticregression__C': [1,5,10]
        }
    }
}

print("done1")

scores = []
best_estimators = {}
import pandas as pd
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model'])
    clf =  GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

print("done2")
    
print("\n")
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)
print("done3")

print("\n")
print(best_estimators)

print("\n")
print("SVM: ",best_estimators['svm'].score(X_test,y_test))
print("Random Forest: ",best_estimators['random_forest'].score(X_test,y_test))
print("logistic regression: ",best_estimators['logistic_regression'].score(X_test,y_test))
best_clf = best_estimators['svm']
print("done4")

print("\n")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_clf.predict(X_test))
print(cm)

print("\n")
print(class_dict)


import joblib 
# Save the model as a pickle in a file 
joblib.dump(best_clf, 'saved_model.pkl') 
print("done pickling")

import json
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))
print("done json file")

