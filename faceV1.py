#import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"persons_name": 1}
with open("face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('rtsp://[ma]%2540[g]:[pa]@[ip]:8001/0/av0')

font = cv2.FONT_HERSHEY_SIMPLEX
color1 = (255,255,255)
stroke = 2
img_item = "my_image.png"

while(True):
    #Frame by frame
    ret, frame = cap.read()
    #print(cap.get(cv2.CAP_PROP_FPS))
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w] #frame size
        roi_color = frame[y:y + h, x:x + w]
        id_, conf = recognizer.predict(roi_gray)

        if conf >= 45:
            #print(id_)
            #print(labels[id_])
            name = labels[id_]
        else:
            name = "UnKnown"

        cv2.putText(frame, name, (x, y), font, 1, color1, stroke, cv2.LINE_AA)
        cv2.imwrite(img_item, roi_color)

        color = (255,0,0) #BGR 0-255
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x,y), (end_cord_x ,end_cord_y ), color, stroke)

    #Display the resulting frame
    cv2.imshow('MotionDetector System', frame)
    if cv2.waitKey(20) or 0xFF == ord('q'):
        break


#Release the capture
cap.release()
cv2.destroyALLWindows()



