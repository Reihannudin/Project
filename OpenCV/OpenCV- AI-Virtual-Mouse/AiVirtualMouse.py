# import library
import cv2
import time
from cv2 import imshow
from cv2 import waitKey
import pyautogui
import os
import numpy as np
import HandsTrackingModule as htm

# runs web cam
cap = cv2.VideoCapture(0)

# set height and weigh cam
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)

frameR = 100 # Frame Reduction

# def pTime
pTime = 0

# def previouse Location
plocX, plocY = 0,0

# def Current Location
clocX , clocY = 0,0

# def landmark
detector = htm.handDetector(maxHands=1)

# def smoothing
smoothing = 10

# def size screen
wScr, hScr = pyautogui.size()
print(wScr, hScr)

while True:

    # 1. find hand landmark
    success, img = cap.read()
    img = detector.findHands(img)
    lmList , bbox = detector.findPosition(img)
    
    # 2. Get the tip if index finger and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        
        # print(x1, y1, x2, y2)
    
        # 3. Check with finger are up
        fingers = detector.fingersUp()
        # print(fingers)
        
        # Boxing room mouse
        cv2.rectangle(img,(frameR, frameR),(wCam-frameR, hCam-frameR),
                      (0,255,0),2)
    
        # 4. Only index finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
           # 5. Convert Coordinate
           x3 = np.interp(x1, (frameR, wCam - frameR), (0,wScr))
           y3 = np.interp(y1, (frameR, hCam - frameR), (0,hScr))
           
           # 6. Smoothen value
           clocX = plocX + (x3 - plocX) / smoothing
           clocY = plocY + (y3 - plocY) / smoothing
           
           # 7. Move mouse
           pyautogui.moveTo(wScr - clocX, clocY)
           cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)#circle shows that we are in moving mode
           plocX, plocY = clocX, clocY
           
        # 8. Both Index and Middle Finger are up: Click Mode
        if fingers[1] == 1 and fingers[2] == 1:
            
           # 9. Find Distance beetween fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            
           # 10. Click mouse if distance is short
            if length < 30:
               cv2.circle(img, (lineInfo[4], lineInfo[5]),
                          15, (0, 255, 0), cv2.FILLED)
               pyautogui.click()
           
    
    
    # 11. Frame rate
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 2)
    # 12. Display
    cv2.imshow("WEB CAM", img)
    if waitKey (1) & 0XFF == ord('q'):
        break   