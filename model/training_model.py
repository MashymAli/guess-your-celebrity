from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cv2 as cv
import pickle
import numpy as np
import pywt   #PyWavelets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

#open dictionary where all paths of cleaned images in dataset is stored
file = "celebrity_file_names_paths_dict.pkl"
fileobj = open(file, 'rb')
celebrity_file_names_dict = pickle.load(fileobj)

# wavelet transform is basically highliting parts of face for recognition
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



#provides 0 till 5 index to each folder basically  {"Angelina Jolie": 0, "Brad Pitt": 1, ......}
class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys(): 
    class_dict[celebrity_name] = count
    count = count + 1
print(class_dict)



X, y = [], []
for celebrity_name, training_files in celebrity_file_names_dict.items(): #iterates through celebrities
    for training_image in training_files: #iterates images of that celebrities
        img = cv.imread(training_image)
        if img is None:
            continue
        scalled_raw_img = cv.resize(img, (32, 32)) #resizing images and getting scaled raw image
        img_har = w2d(img,'db1',5)  #wavelet transform image
        scalled_img_har = cv.resize(img_har, (32, 32)) #getting scaled wavelet image
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1))) #stacking raw and wavelet transformed image  32*32*3 means rgb image and 2*32 meand grayscale image
        X.append(combined_img)
        y.append(class_dict[celebrity_name]) #index of celebrity name from saved dict

#Checking no of images in X
X = np.array(X).reshape(len(X),4096).astype(float)
print(X.shape)  #prints (2212, 4096) which means (no.of images, pic is represented as 1d array [32*32*3 + 32*32])

#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) #random state=0 means same data for test and train is used across different executions


# TRYING DIFFERENT ALGOS FOR ACCURACY

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


scores = []
best_estimators = {}
import pandas as pd
for algo, mp in model_params.items():
    pipe = make_pipeline(StandardScaler(), mp['model']) #pipeline is flow of work
    clf =  GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False) #classifier
    clf.fit(X_train, y_train) # fit the pipeline to the training data
    scores.append({
        'model': algo,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    best_estimators[algo] = clf.best_estimator_

    
print("\n")
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)

print(best_estimators)


print("\n Accuracy with test data: ")
print("SVM: ",best_estimators['svm'].score(X_test,y_test))
print("Random Forest: ",best_estimators['random_forest'].score(X_test,y_test))
print("logistic regression: ",best_estimators['logistic_regression'].score(X_test,y_test))
best_clf = best_estimators['logistic_regression']


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, best_clf.predict(X_test))
print(cm)

print(class_dict)

#saving model that we have trained above
file = "saved_model.pkl"
fileobj = open(file, 'wb')
pickle.dump(best_clf, fileobj)
fileobj.close()

import json
with open("class_dictionary.json","w") as f:
    f.write(json.dumps(class_dict))
print("done json file")