import cv2 as cv
import os, shutil
import pickle



#calling face and eye features from haarcascades, so we can crop the pictures based on eye features
face_cascade = cv.CascadeClassifier('model/opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('model/opencv/haarcascades/haarcascade_eye.xml')


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
            return roi_color   #roi = region of interest


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