import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer/trainingData.yml")
id=0
#font=cv2.InitFont(cv2.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
fontface=cv2.FONT_HERSHEY_SIMPLEX
fontscale=2
fontcolor=(255,0,0)
fontsize=4
while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
          cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
          id,conf=rec.predict(gray[y:y+h,x:x+w])
          if(id==1):
              id="Vaibhav"
          if(id==2):
               id="Narendra Modi"
          if(id==3):
               id="Rounak"
          if (id==4):
                id="Harsh"
          if (id==5):
                id="Jiten"
          if (id==8):
              id="Aditya"
          cv2.putText(img,str(id),(x,y+h),fontface,fontscale,fontcolor,fontsize);
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
                             



cam.release()
cv2.destroyAllWindows()
