import os
import cv2 as cv 
import numpy as np 

people = ['Christian Bale','Mr White','Ossy Osbourne','Ryan Gosling']

haarCascade = cv.CascadeClassifier('OpenCV\Face Detection with Haar Cascades\haar_face.xml')
features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join('OpenCV\Face Recognizer\Training Photos',person)
        label = people.index(person)
        
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_arr = cv.imread(img_path)
            gray = cv.cvtColor(img_arr,cv.COLOR_BGR2GRAY)
            faces_rect = haarCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h,x:x+w]
                features.append(faces_roi)
                labels.append(label)
    pass

create_train()
print("--Training Done-----------------")
features = np.array(features,dtype='object')
labels = np.array(labels)
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train The Recognizer on the Features List an the Labels
face_recognizer.train(features,labels)
face_recognizer.save('face_train.yml')
np.save('features.npy',features)
np.save('labels.npy',labels)
