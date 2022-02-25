# import libraries
import cv2 
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320
confThreshold = 0.5
nmsthreshold = 0.3

# make data in coco names to be array
classFile = 'coco.names'
classNames = []
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n') 
# print(classNames)
# print(len(classNames))

# configure file
modelConfiguration =  'yolov3.cfg'
modelWeights = 'yolov3.weights' 

# create network
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# mke func find object
def findObjects(outputs, img):
    height, width, channels = img.shape
    bound_box = []
    classIds = []
    confident_val = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * width), int(det[3] * height)
                x, y = int((det[0]* width) - w/2), int((det[1]* height) - h/2)
                bound_box.append([x, y, w, h])
                classIds.append(classId)
                confident_val.append(float(confidence))
        
    # print(len(bound_box))
    
        indices = cv2.dnn.NMSBoxes(bound_box, confident_val, confThreshold, nmsthreshold) # non max supression
        # print(indices)
         
        for i in indices:
            i = i
            # print("i", i)
            box = bound_box[i]
            # print("box", box)
            x,y,w,h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x,y), (x+w,y+h),color=(0, 255, 0), thickness=2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confident_val[i]*100)}%', 
                       (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0), 2) 
    
    
while True:
    # read the frame
    success , img = cap.read()
    
    # make blob
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT,whT),[0,0,0],1, crop=False)
    net.setInput(blob)
    
    # get the names of the layers
    layersName = net.getLayerNames() 
    # print(layersName)
    
    # Get the ouput layers
    outputNames = [layersName[i-1] for i in net.getUnconnectedOutLayers()] # -1 is because the first layer is layer 0
    # print(outputNames)
    
    # get the output of the network
    outputs =  net.forward(outputNames) 
    
    # print lenght of outputs
    # print(len(outputs))
    
    # print type List  
    #print(type(outputs[0].shape)) 
    
    # print Metrics
    #print(outputs[0].shape) 
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    
    # print(outputs[0][0])
    
    # call fungtion find object
    findObjects(outputs,img)    
    
    # activate web Cam
    cv2.imshow("img",img) # show the frame
    cv2.waitKey(1) # wait for 1 ms