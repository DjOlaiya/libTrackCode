from __future__ import print_function
import argparse
import random as rng
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imutils
from imutils import contours, perspective
from scipy.spatial import distance as dist


def midpoint(ptA,ptB):
    return ((ptA[0]+ptB[0])*0.5, (ptA[1]+ptB[1])*0.5)

def drawLine():
    cv.namedWindow("preview")
    vc = cv.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv.waitKey(20)
        if key == 27: # exit on ESC
            break
        else:
            cv.line(img=frame, pt1=(200, 10), pt2=(100, 10), color=(255, 0, 0), thickness=2, lineType=8, shift=0)

    vc.release()
    cv.destroyWindow("preview")   

def resizeWindow(img,height=1600):
    ratio = float(img.shape[1]/img.shape[0])
    width = height/ratio
    image = cv.resize(img,(int(height),int(width)))
    return image


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", help="image path")
args = vars(ap.parse_args())

#learning edge detection.
img = cv.imread(args['image'])
# cv.imshow('img', img)
# img = cv

#always check num channels and src type
# print('img channels ', img.shape,'\n','imge type',img.dtype)
kernel = np.ones((5,5),np.uint8)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gauss = cv.GaussianBlur(gray,(7,7),0)
# bilateral = cv.bilateralFilter(gray,9,75,75) worse edge detection
#lets see imgs side by side og and blur
# openimg = cv.morphologyEx(bilateral,cv.MORPH_OPEN,kernel)
closeimg = cv.morphologyEx(gauss,cv.MORPH_CLOSE,kernel)
# closeimg2 = cv.morphologyEx(bilateral,cv.MORPH_CLOSE,kernel)
inp = gauss
# inp2 = bilateral
edges = cv.Canny(inp,50,100)
print("step 1 find edges {}".format(edges))
cv.imshow("edge detected",edges)
# edges2 = cv.Canny(inp2,50,100)
edges = cv.dilate(edges, None, iterations=1)
edges = cv.erode(edges,None,iterations=1)

lines = cv.HoughLinesP(edges,1,np.pi/180,60,np.array([]),50,5)
for line in lines:
    for x1, y1, x2, y2 in line:
        cv.line(inp,(x1,y1),(x2,y2),(255,0,0),6)
        cv.line(edges,(x1,y1),(x2,y2),(255,0,0),3)
# edges2 = cv.dilate(edges2, None, iterations=1)
# edges2 = cv.erode(edges2,None,iterations=1)
adjacentImg = np.concatenate((inp,edges),axis=1)
# adjacentImg2 = np.concatenate((inp2,edges2),axis=1)
resizedImg = resizeWindow(adjacentImg)
# resizedImg2 = resizeWindow(adjacentImg2)
cv.imshow('Edge Detection 1.0', resizedImg)
cv.imwrite('FailedEdgeDetection.png', resizedImg)
# cv.imshow('compare img bilateral', resizedImg2)
cv.imshow("original", img)
cv.imshow("edges",edges)
cv.waitKey(0)
# cnt = cv.findContours(edges.copy(),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
# cnt = imutils.grab_contours(cnt)

# (cnt,_)= contours.sort_contours(cnt)
# ppM = None

# for c in cnt:

#     if cv.contourArea(c) < 12000:
#         continue

#     og = img.copy()
#     box = cv.minAreaRect(c)
#     box = cv.cv.boxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
#     box = np.array(box,dtype='int')


#     box = perspective.order_points(box)
#     cv.drawContours(og,[box.astype('int')],-1,(0,255,0),2)

#     for (x,y) in box:
#         cv.circle(og,(int(x),int(y)),5,(0,0,255),-1)


#     (tl,tr,br,bl) = box
#     print((tl,tr))
#     print((bl,br))
#     tltrX,tltrY = midpoint(tl,tr)
#     blbrX,blbrY = midpoint(bl,br)
#     tlblX,tlblY = midpoint(tl,bl)
#     trbrX,trbrY = midpoint(tr,br)
#     print("here are tltrX{} and tltrY{}".format(tltrX,tltrY))
#     print("here are blbrX{} and blbrY{}".format(blbrX,blbrY))
#     print("here are tlblX{} and tlblY{}".format(tlblX,tlblY))
#     print("here are trbrX{} and trbrY{}".format(trbrX,trbrY))
#     cv.circle(og,(int(tltrX),int(tltrY)),5,(255,0,0),-1)
#     cv.circle(og,(int(blbrX),int(blbrY)),5,(255,0,0),-1)
#     cv.circle(og,(int(tlblX),int(tlblY)),5,(255,0,0),-1)
#     cv.circle(og,(int(trbrX),int(trbrY)),5,(255,0,0),-1)

#     cv.line(og,(int(tltrX),int(tltrY)),(int(blbrX),int(blbrY)),(255,0,255),2)
#     cv.line(og,(int(tlblX),int(tlblY)),(int(trbrX),int(trbrY)),(255,0,255),2)

#     dA = dist.euclidean((tltrX,tltrY),(blbrX,blbrY))
#     print("here is dA {}".format(dA))
#     dB = dist.euclidean((tlblX,tlblY),(trbrX,trbrY))
#     print("here are coordinate for tlblX and tlblY {}".format((tlblX,tlblY)))
#     print("here are coordinate for trbrX and trbrY {}".format((trbrX,trbrY)))
#     print("here is db {}".format(dB))
#     if ppM is None:
#         ppM = dB/3.37
#         print("here is ppm {}".format(ppM))
#     dimA = dA/ppM
#     dimB = dB/ ppM

#     cv.putText(og,"{:.1f}in".format(dimA),(int(tltrX -15),int(tltrY-10)),cv.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
#     cv.putText(og,"{:.1f}in".format(dimB),(int(trbrX +1),int(trbrY)),cv.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)

# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()
# cv.waitKey()

rng.seed(12345)
def thresh_callback(val):
    threshold = val
    
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    
    
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
    
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours_poly, i, color)
        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    
    
    cv.imshow('Contours', drawing)
    
# parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes and circles for contours tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
# args = parser.parse_args()
# src = cv.imread(cv.samples.findFile(args.input))
# if src is None:
#     print('Could not open or find the image:', args.input)
#     exit(0)
# # Convert image to gray and blur it
# src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# src_gray = cv.blur(src_gray, (3,3))
# source_window = 'Source'
# cv.namedWindow(source_window)
# cv.imshow(source_window, src)
# max_thresh = 255
# thresh = 100 # initial threshold
# cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
# thresh_callback(thresh)
# cv.waitKey()