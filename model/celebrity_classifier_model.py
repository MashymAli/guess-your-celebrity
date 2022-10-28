import cv2 as cv
#import matplotlib
#from matplotlib import pyplot as plt
import os, shutil
import pickle


#reading original image
#img = cv.imread('model/test/aamir1.jpeg')
#print(img.shape)
#plt.imshow(img)
#plt.show()

#converting orignal image to grayscale
#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#print(gray.shape)
#print(gray)
#plt.imshow(gray,cmap='gray')
#plt.show()

#calling face and eye features from haarcascades, so we can crop the pictures based on eye features
face_cascade = cv.CascadeClassifier('model/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('model/opencv/haarcascades/haarcascade_eye.xml')

#faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#print(faces)  #return [220  31 293 293]  means x,y width,height where face lies in picture 


#(x,y,w,h) = faces[0]  
#print(x,y,w,h)  #x-axis , y-axis , width , height

#face_img = cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)  #prints rectangle around the face
#plt.imshow(face_img)
#plt.show()

#function that will give cropped images when it can detect two eyes in it
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
#print(img_dirs)  #print array/list of all folder names in dataset folder


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


#pickling this dictionary so we can access this into our training_model file
file = "celebrity_file_names_paths_dict.pkl" #this dict has paths to images in cropped folder
fileobj = open(file, 'wb')
pickle.dump(celebrity_file_names_dict, fileobj)
fileobj.close()