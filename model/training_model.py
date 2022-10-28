from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import cv2 as cv
import pickle
import numpy as np
import pywt



file = "celebrity_file_names_paths_dict.pkl"
fileobj = open(file, 'rb')
celebrity_file_names_dict = pickle.load(fileobj)

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




class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():  #provides 0 till 99 index to each folder basically
    class_dict[celebrity_name] = count
    count = count + 1
#print(class_dict)



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

X = np.array(X).reshape(len(X),4096).astype(float)
print(X.shape)  #prints (2212, 4096) which means (no.of images, pic is represented as 1d array)

# THIS IS NOT WORKING (▱˘︹˘▱)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 1000))]) #creating svm model
pipe.fit(X_train, y_train)  #training model created above
print("Score: ",pipe.score(X_test, y_test))
print("Total images tested: ",len(X_test))

#We cannot use SVC anymore as its not giving much accuracy (▱˘︹˘▱). 
#I have searched a bit and found that svc is not much suitable for large datasets 😭😭

#print("\n")
#print(classification_report(y_test, pipe.predict(X_test)))
#
#from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import GridSearchCV
#
#model_params = {
#    'svm': {
#        'model': svm.SVC(gamma='auto',probability=True),
#        'params' : {
#            'svc__C': [1,10,100,1000],
#            'svc__kernel': ['rbf','linear']
#        }  
#    },
#    'random_forest': {
#        'model': RandomForestClassifier(),
#        'params' : {
#            'randomforestclassifier__n_estimators': [1,5,10]
#        }
#    },
#    'logistic_regression' : {
#        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
#        'params': {
#            'logisticregression__C': [1,5,10]
#        }
#    }
#}
#
#print("done1")
#
#scores = []
#best_estimators = {}
#import pandas as pd
#for algo, mp in model_params.items():
#    pipe = make_pipeline(StandardScaler(), mp['model'])
#    clf =  GridSearchCV(pipe, mp['params'], cv=5, return_train_score=False)
#    clf.fit(X_train, y_train)
#    scores.append({
#        'model': algo,
#        'best_score': clf.best_score_,
#        'best_params': clf.best_params_
#    })
#    best_estimators[algo] = clf.best_estimator_
#
#print("done2")
#    
#print("\n")
#df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
#print(df)
#print("done3")
#
#print("\n")
#print(best_estimators)
#
#print("\n")
#print("SVM: ",best_estimators['svm'].score(X_test,y_test))
#print("Random Forest: ",best_estimators['random_forest'].score(X_test,y_test))
#print("logistic regression: ",best_estimators['logistic_regression'].score(X_test,y_test))
#best_clf = best_estimators['svm']
#print("done4")
#
#print("\n")
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, best_clf.predict(X_test))
#print(cm)
#
#print("\n")
#print(class_dict)
#
#
#import joblib 
## Save the model as a pickle in a file 
#joblib.dump(best_clf, 'saved_model.pkl') 
#print("done pickling")
#
#import json
#with open("class_dictionary.json","w") as f:
#    f.write(json.dumps(class_dict))
#print("done json file")