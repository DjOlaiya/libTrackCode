import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import imutils
from imutils import perspective, contours
import time
import os
#cmd line 
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", help="image path")
args = vars(ap.parse_args())

face_cascade = cv.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
# cv2_base_dir = os.path.dirname(os.path.abspath(cv.__file__))
# haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

#get img
image = cv.imread(args['image'])
gray = cv.cvtColor(image,cv.COLOR_RGB2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv.rectangle(image,(x,y),(x+w*1.2,y+h*1.1),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

cv.imshow('img',image)
cv.waitKey(0)
cv.destroyAllWindows()
#foreground extraction not effective currently. What if I detect face first?
# mask = np.zeros(image.shape[:2], np.uint8)
# bgdModel = np.zeros((1,65),np.float64)
# fgdModel = np.zeros((1,65),np.float64)

# rect = (550,150,1550,950)
# cv.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)

# mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
# image = image*mask2[:,:,np.newaxis]

# plt.imshow(image),plt.colorbar(),plt.show()

# gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
# edge = cv.Canny(gray,100,300)
# print("STEP 1. Get Image It works")

# #find and outline contours
# cnt,h = cv.findContours(edge.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# # cnt = imutils.grab_contours(cnt)
# cnt = sorted(cnt, key=cv.contourArea, reverse=True)
# print("there are {} contours in list".format(len(cnt)))
# #args: img, contour list, index, color, thickness
# # (cnt,_) = contours.sort_contours(cnt)
# itera = 0
# outloop = 0
# # for c in cnt:
# #     print("outer loop counter is {}".format(outloop))
# #     outloop+=1
# #     if cv.contourArea(c) < 10:
# #         print("this was discarded {}".format(itera))
# #         itera +=1
# #         continue

# #     og = image.copy()
#     # box = cv.minAreaRect(c)
#     # box = cv.cv.boxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
#     # box = np.array(box,dtype='int')
#     # cv.drawContours(og,c,-1,(0,255,0), thickness=4)
#     # box = perspective.order_points(box)
#     # cv.drawContours(og,[box.astype('int')],-1,(0,0,255),4)
#     # cv.imshow("contours drawn detected",og)
#     # cv.waitKey(0)
#     # for (x,y) in box:
#     #     cv.circle(og,(int(x),int(y)),5,(0,0,255),-1)
# # cont = cnt
# # for c in cnt:
# #     cv.drawContours(image,c,-1,(0,255,0), thickness=4)
# #     cv.imshow("image",image)
# #     cv.waitKey(0)

# cv.drawContours(image,cnt[15],-1,(0,255,0), thickness=4)
# print("STEP 2. Draw Contours works")
# cv.imshow("image",image)
# cv.waitKey(0)
