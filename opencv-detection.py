#Author : Saket Srivastava
#Model trained using CNN


##Importing Libraries
import cv2
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import Image

##Importing our trained CNN model
new_model = tf.keras.models.load_model('model.tf')

##Importing our Haar Cascade feature detector for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

## Detect funtion to detect face and then predict weather person is wearing mask or not
def detect(grey , frame):
    img = Image.fromarray(frame)
    img.save('1.png')
    x = image.load_img('1.png',target_size=(150,150))
    x = image.img_to_array(x)
    x = np.expand_dims(x,axis = 0)
    result = new_model.predict(x)
    if result[0][0] == 1:
        text = "Mask Detected"
    else:
        text = "Mask Not Detected"
    faces = face_cascade.detectMultiScale(grey , 1.3, 5)
    for (x,y , w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1,cv2.LINE_AA)
    return frame    

## OpenCV loop

video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    print(frame)
    grey = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    canvas = detect(grey,frame)
    cv2.namedWindow('Mask',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mask', 1600,800)
    cv2.imshow('Mask',canvas)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()    