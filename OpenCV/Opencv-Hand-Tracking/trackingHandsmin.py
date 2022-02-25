# import library
import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) # 0 is the id of the camera

mpHands = mp.solutions.hands # mpHands is the class of the solution
hands = mpHands.Hands() # hands is the instance of the class
mpDraw = mp.solutions.drawing_utils # npDraw is the class of the solution
ctime = 0
ptime = 0


while True:
    
    success , img = cap.read() # read the frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB 
    results = hands.process(imgRGB) # process the frame
    
    # print(result.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id  , lm in enumerate(handLms.landmark):
                #print(id , lm)
                h,w,c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                if id == 4:
                    cv2.circle(img,(cx, cy), 15, (0,255,0), cv2.FILLED)
     
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
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