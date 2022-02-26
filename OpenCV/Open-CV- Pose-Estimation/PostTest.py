import cv2
import time 
import mediapipe as mp
import PoseModule as pm

cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector()
while True:
    success , img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPositions(img)
    print(lmList)
        # make fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    cv2.imshow("Web cam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break