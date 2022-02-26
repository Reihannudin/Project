# import library
import time
from turtle import color
from unittest import result
import cv2
from cv2 import waitKey
from cv2 import circle
import mediapipe as mp

# ruining the web cam
cap = cv2.VideoCapture(0)
pTime =  0

# def facemesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

while True:
    success , img = cap.read()
    
    # BGR to RGB conversion
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = faceMesh.process(imgRGB)
    
    # show faceMesh
    if result.multi_face_landmarks:
        for faceLms in result.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            # chose the poiny to draw
            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id, x, y)
    
    # make fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:  {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    
    
    cv2.imshow('Web Cam', img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
       break