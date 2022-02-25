# import library
import cv2 
import mediapipe as mp
import time
import handsTrackingModule as htm

ctime = 0
ptime = 0
cap = cv2.VideoCapture(0) # 0 is the id of the camera
detector = htm.handDetector()
  
while True:
    success , img = cap.read() # read the frame
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
      
    if len(lmList) != 0:
        print(lmList[4])
      
    # make FPS report
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
      
   
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0,255,0),3)
              
    # runs Web Cam
    cv2.imshow("Web Cam", img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
       break