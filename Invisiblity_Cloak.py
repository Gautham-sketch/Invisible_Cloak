import cv2
from matplotlib.pyplot import axis
import numpy as np
import time

cap = cv2.VideoCapture(0)
video_writer = cv2.VideoWriter_fourcc(*"XVID")
output_file = cv2.VideoWriter('Output.avi',video_writer,24,(640,450))
time.sleep(2)

bg = 0
for i in range(60):
    r1,bg = cap.read()
bg = np.flip(bg,axis=1)

while(cap.isOpened()):
    ret,img = cap.read()
    if not ret:
        break
    img = np.flip(img,axis=1)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,120,50])
    upper_red = np.array([10,255,255])
    mask1 = cv2.inRange(hsv,lower_red,upper_red)
    lower_red = np.array([170,120,50])
    upper_red = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,lower_red,upper_red)
    mask = mask1 + mask2
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    mask2 = cv2.bitwise_not(mask)

    res1 = cv2.bitwise_and(img,img,mask = mask2)
    res2 = cv2.bitwise_and(bg,bg,mask = mask)
    final_output = cv2.addWeighted(res1,1,res2,1,0)
    output_file.write(final_output)
    cv2.imshow("Invisible",final_output)