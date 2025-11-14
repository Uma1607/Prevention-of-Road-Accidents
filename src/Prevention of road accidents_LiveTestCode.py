# Download Anaconda based on OS (Windows/MAC/Linux)
# Anaconda Navigator 
# Jupyter Notebook
# check update Anaconda 
# $conda update conda
# $conda update anaconda
# Create a new environment in Anaconda promt and install Dlib 
# to create environmnet $conda create -n env_name python=version number
# to remove environament $conda env remove -n env_name
# activating environment $conda activate env_name
# deactivating environment $conda deactivate
# Run Anaconda prompt in RUN AS ADMINISTRATOR


# $pip install tensorflow (if using gpu # pip install tensorflow-gpu)
import tensorflow as tf
# $pip install opencv-contrib-python 
import cv2
import os
# $pip install matplotlib
import matplotlib.pyplot as plt
# $pip install numpy
import numpy as np
import winsound
# for Anaconda - $conda install -c conda-forge dlib
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils


#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Users/uma/shape_predictor_68_face_landmarks.dat')

#status marking for current state
sleep = 0
drowsy = 0
active = 0
status=""
frequency = 2500
duration = 2000

frequency2 = 2500
duration2 = 5000
color=(0,0,0)

def compute(ptA,ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a,b,c,d,e,f):
    up = compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)

    #Checking if it is blinked
    if(ratio>0.25):
        return 2
    elif(ratio>0.21 and ratio<=0.25):
        return 1
    else:
        return 0
    
    
path = "haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN


#setting rectangle background to white
rectangle_bgr = (255,255,255)
#making a black image
img = np.zeros((500,500))
#setting some text
text = "Text in Box!!"
new_model = tf.keras.models.load_model("C:/Yawn_NoYawn/yawnDetection.h5")

#getting width and height of the text box
(text_width,text_height) = cv2.getTextSize(text,font,fontScale = font_scale, thickness=1)[0]

#setting the start position for text
text_offset_x = 10
text_offset_y = img.shape[0] - 25

#making co-ordinates of the box with padding of two pixels
box_coords = ((text_offset_x,text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale = font_scale, color=(0, 0, 0), thickness=1)

cap = cv2.VideoCapture(1)

# checking whether web cam is open or not 
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret,frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for x,y,w,h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess)==0:
            print("face not detected!!")
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi=roi_color[ey: ey+eh, ex: ex+ew]
        
    final_image = cv2.resize(face_roi,(224,224))
    final_image = np.expand_dims(final_image,axis=0)
    final_image = final_image/255  # normalising data
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    predictions = new_model.predict(final_image)
    score = tf.nn.softmax(predictions[0])
    
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.5
    
    faces = detector(gray)
    #detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        #The numbers are actually the landmarks which will show eye
        left_blink = blinked(landmarks[36],landmarks[37], 
            landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43], 
            landmarks[44], landmarks[47], landmarks[46], landmarks[45])
  
        
    if(np.argmin(score)==1):
        status ="Yawn"
        x1,y1,w1,h1 = 0,0,175,75
        # drawing black background rectangle
        cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
        cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(frame, status, (100, 150),font, 3, (255,255,255), 2, cv2.LINE_4)
        winsound.Beep(2500,2000)
    
    elif(left_blink==0 or right_blink==0):
    #Now judge what to do for the eye blinks
        sleep+=1
        drowsy=0
        active=0
        if(sleep>6):
            status="Sleepy"
            x1,y1,w1,h1 = 0,0,175,75
            # drawing black background rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
            cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, status, (100, 150),font, 3, (255,255,255), 2, cv2.LINE_4)
            winsound.Beep(2500,2000)
                
    elif(left_blink==1 or right_blink==1):
        sleep=0
        active=0
        drowsy+=1
        if(drowsy>6):
            status="Drowsy!"
            x1,y1,w1,h1 = 0,0,175,75
            # drawing black background rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
            cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, status, (100, 150),font, 3, (255,255,255), 2, cv2.LINE_4)
            winsound.Beep(2500,2000)
    
    elif (np.argmin(score)==0):
            status ="Active"
            x1,y1,w1,h1 = 0,0,175,75
            # drawing black background rectangle
            cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
            cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, status, (100, 150),font, 3, (255,255,255), 2, cv2.LINE_4)
            
     
    for n in range(0, 68):
        (x,y) = landmarks[n]
        cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        
    cv2.imshow('Face Emotion Recognition',frame)
    cv2.imshow("Result of detector", face_frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
