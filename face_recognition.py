import numpy as np 
import cv2 as cv 

people = ['Christian Bale','Mr White','Ossy Osbourne','Ryan Gosling']

 
haarCascade = cv.CascadeClassifier('OpenCV\Face Detection with Haar Cascades\haar_face.xml')
#features = np.load('features.npy')
#labels = np.load('labels.npy')
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_train.yml')

img = cv.imread('OpenCV\Face Recognizer\Validation\MrWhite1.webp')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Bale',gray)

# Detect The Image

faces_rect = haarCascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,1.0,(0,0,255),thickness = 2,)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),thickness=2)
cv.imshow('Detected Face',img)
cv.waitKey(0)