import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)
print(kernel)

path ="C:/Users/Hari Prasad/Pictures/Screenshots/Screenshot_20230202_192521.png"
img = cv2.imread(path)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
cv2.imshow("Lena",img)
cv2.imshow("img Blur",imgBlur)  
cv2.waitkey(0)
