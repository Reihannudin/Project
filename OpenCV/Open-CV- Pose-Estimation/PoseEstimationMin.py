from unittest import result
import cv2
from cv2 import waitKey
import mediapipe as mp
import time

# draw_keypoints_on_image
mpDraw = mp.solutions.drawing_utils

# make Pose 
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture(0)
pTime = 0

while True:
    success , img = cap.read()
    
    # converter image bgr 2 rgb
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)
    print(result.pose_landmarks)
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id , lm in enumerate(result.pose_landmarks.landmark):
            h,w , c = img.shape
            print(id, lm)
            cx , cy = int(lm.x*w) , int(lm.y*h)
            cv2.circle(img, (cx, cy), 3 , (0,255,0), cv2.FILLED)
            
    
    # make fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    cv2.imshow("Web cam", img)
    if waitKey(1) & 0xFF == ord('q'):
        break