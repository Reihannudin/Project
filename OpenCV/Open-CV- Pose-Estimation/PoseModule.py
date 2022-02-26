import cv2
import time
import mediapipe as mp

class poseDetector():
    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackingCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        
        # draw_keypoints_on_image
        self.mpDraw = mp.solutions.drawing_utils
        
        # make Pose 
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackingCon)
        
        # find te pose
    def findPose(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result =self.pose.process(imgRGB)
        if self.result.pose_landmarks:
           if draw:
               self.mpDraw.draw_landmarks(img, self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
               
        return img
    
    # find the points
    def findPositions(self, img, draw=True):
        lmList = []
        if self.result.pose_landmarks:
           for id , lm in enumerate(self.result.pose_landmarks.landmark):
               h,w , c = img.shape
               #print(id, lm)
               cx , cy = int(lm.x*w) , int(lm.y*h)
               lmList.append([id, cx, cy])
               if draw:
                  cv2.circle(img, (cx, cy), 3 , (0,255,0), cv2.FILLED)
               
        return lmList

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
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

if __name__ == '__main__':
    main()