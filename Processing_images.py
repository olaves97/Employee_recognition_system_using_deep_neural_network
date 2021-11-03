import cv2
import os

imagePath = 'C:/Users/User/Desktop/Employee1_before'
changed_imagePath = 'C:/Users/User/Desktop/Employee1'

myList = os.listdir(imagePath)
print(myList)

for cl in myList:
    image = cv2.imread(f'{imagePath}/{cl}', cv2.IMREAD_GRAYSCALE)
    new_image = cv2.resize(image, (112, 112))
    new_array = new_image.reshape(-1, 112, 112, 1)
    cv2.imwrite(f'{changed_imagePath}/{cl}', new_image)
    cv2.waitKey(0)
