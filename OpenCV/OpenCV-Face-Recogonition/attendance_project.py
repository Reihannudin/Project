# Import Library
from matplotlib import image
import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime


# mengambil data dari folder dataset
path = 'dataset'
images= [] 
classNames = []
myList = os.listdir(path)
print(myList)  # menampilkan nama file yang ada di dalam folder dataset

# menghapus format
for cli in myList:
    curimg = cv2.imread(f'{path}/{cli}') # baca file
    images.append(curimg)
    classNames.append(os.path.splitext(cli)[0])
print(classNames)

# meng encodingkan data images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# membuat fungsi attendance ke  dalam file csv
def markAttendence(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in  myDataList:
            entery = line.split(',')
            nameList.append(entery[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            

# markAttendence("Reihan")


encodeList = findEncodings(images)
# print(len(encodeList))
print("Encoding Complete")

# add camera
cap = cv2.VideoCapture(0)

while True:
    succes, img = cap.read() # baca camera
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCutFrame = face_recognition.face_locations(imgS)
    encodesCutFrame = face_recognition.face_encodings(imgS, facesCutFrame)
    
    for encodeFace, faceloc in zip(encodesCutFrame, facesCutFrame):
        matches = face_recognition.compare_faces(encodeList, encodeFace)
        faceDis = face_recognition.face_distance(encodeList, encodeFace)
        print(faceDis)
        
        # memberikan labels
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            
            # make green box
            y1,x2,y2,x1 = faceloc # get location
            y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2,y2) ,(0,255,0),cv2.FILLED)
            cv2.putText(img,name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
            # memanggil fungsi menambahkan nama dan waktu ke csv
            markAttendence(name)
            
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

# Load Image
# img_face = face_recognition.load_image_file('dataset/reihan (11).jpg')    # Load Image
# img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)                   # Convert Image to RGB
# 
# img_face2 = face_recognition.load_image_file('dataset/reihan (5).jpg')    # Load Image
# img_face2 = cv2.cvtColor(img_face2, cv2.COLOR_BGR2RGB)                   # Convert Image to RGB
