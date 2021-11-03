import cv2
import os


imagePath = 'C:/Users/User/Desktop/Employee1'
cascPath = "haarcascade_frontalface_default.xml"
myList = os.listdir(imagePath)


#Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

for cl in myList:
    image = cv2.imread(f'{imagePath}/{cl}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.3, 4)

    #Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(image, (x, y), (x+w, y+h), (128, 128, 128), 0)
        faces = image[y:y + h, x:x + w]
        #cv2.imshow("Photo of the person", faces)
        faces = cv2.resize(faces, (112, 112))
        cv2.imwrite(f'{imagePath}/{cl}', faces)
        cv2.waitKey(0)



