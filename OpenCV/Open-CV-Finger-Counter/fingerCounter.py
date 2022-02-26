import cv2
import os
import mediapipe as mp
import time
import HandsTrackingModule as htm

# set height and width of the image
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# add file in Finger_images to list
folderPath = 'Finger_Images'
myList = os.listdir(folderPath)
#print(myList)

overlay = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlay.append(image)

# print(len(overlay))
pTime = 0
tipIds = [4,8,12,16,20]
detector = htm.handDetector(detectionConf=0.75)

while True:
    success , img = cap.read()
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)
    
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
                
        # 4 Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        #print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)        
            
        #if lmList[8][2] < lmList[6][2]:
        #    print("Index Finger Open")    
    
    
        #h, w, c, = overlay[totalFingers - 1].shape
        img[0:200, 0:200] = overlay[totalFingers - 1]
        
        cv2.rectangle(img,(20,255),(170,425),(255,255,255),cv2.FILLED)
        cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,45,4),25)
    
    # make fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime =cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),3)
    
    cv2.imshow("Web cam", img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break