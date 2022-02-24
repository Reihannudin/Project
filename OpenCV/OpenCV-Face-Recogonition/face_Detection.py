# Import Library
import numpy as np
import cv2
import face_recognition

# Load Image
img_face = face_recognition.load_image_file('dataset/reihan (11).jpg')    # Load Image
img_face = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)                   # Convert Image to RGB

img_face2 = face_recognition.load_image_file('dataset/reihan (5).jpg')    # Load Image
img_face2 = cv2.cvtColor(img_face2, cv2.COLOR_BGR2RGB)                   # Convert Image to RGB


faceloc = face_recognition.face_locations(img_face)[0]                # Get Face Location
encode_face1 = face_recognition.face_encodings(img_face)[0] # Get Encoding
cv2.rectangle(img_face, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (0, 255, 0), 2) # Draw Rectangle

faceloc2 = face_recognition.face_locations(img_face2)[0]                # Get Face Location
encode_face2 = face_recognition.face_encodings(img_face2)[0] # Get Encoding
cv2.rectangle(img_face2, (faceloc2[3], faceloc2[0]), (faceloc2[1], faceloc2[2]), (0, 255, 0), 2) # Draw Rectangle 

# detecting 2 photo
result =face_recognition.compare_faces([encode_face1], encode_face2)

# face distance compare_faces
faceDis = face_recognition.face_distance([encode_face1], encode_face2)

print(result, faceDis)

# add Distance in image
cv2.putText(img_face, f"{result}: {round(faceDis[0],2)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("foto me 1", img_face)
cv2.imshow("foto me 2", img_face2)
cv2.waitKey(0)
