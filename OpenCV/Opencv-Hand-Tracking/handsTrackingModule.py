# import library
import cv2 
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode = False, maxHands = 2, modelC=1, detectionConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf
        
        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelC,self.detectionConf,self.trackingConf) 
        self.mpDraw = mp.solutions.drawing_utils 


    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.results = self.hands.process(imgRGB) 
        # print(result.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img
         
                   
    def findPosition(self,img, handNo=0, draw=True):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id  , lm in enumerate(myHand.landmark):
                    #print(id , lm)
                    h,w,c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    # print(id,cx,cy)
                    lmList.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img,(cx, cy), 7, (0,255,0), cv2.FILLED)

        return lmList
   
def main():
    ctime = 0
    ptime = 0
    cap = cv2.VideoCapture(0) # 0 is the id of the camera
    detector = handDetector()
    
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


if __name__ == "__main__":
    main()