from __future__ import print_function
import argparse
import random as rng
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imutils
from imutils import contours, perspective
from scipy.spatial import distance as dist
from processfile import *
"""
TO DO
use threshold to ignore the background images
"""


def midpoint(ptA, ptB):
    return ((ptA[0]+ptB[0])*0.5, (ptA[1]+ptB[1])*0.5)


def drawLine():
    cv.namedWindow("preview")
    vc = cv.VideoCapture(0)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        cv.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv.waitKey(20)
        if key == 27:  # exit on ESC
            break
        else:
            cv.line(img=frame, pt1=(200, 10), pt2=(100, 10), color=(
                255, 0, 0), thickness=2, lineType=8, shift=0)

    vc.release()
    cv.destroyWindow("preview")


def resizeWindow(img, height=800):
    ratio = float(img.shape[1]/img.shape[0])
    width = height/ratio
    image = cv.resize(img, (int(height), int(width)))
    return image


# ap = argparse.ArgumentParser()
# ap.add_argument("-i","--image", help="image path")
# args = vars(ap.parse_args())

# #learning edge detection.

# img = cv.imread(args['image'])
# canvas = np.zeros(img.shape,np.uint8)
# res = imutils.resize(img,800)
# #STEP 1 #####################
# # cv.imshow('img', res)
# # cv.waitKey(0)
# print("step 0 image resize SUCCESS")
# #always check num channels and src type
# # kernel = np.ones((5,5),np.uint8)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray2 = gray.copy()
# gauss = cv.GaussianBlur(gray,(5,5),0)
# # bilateral = cv.bilateralFilter(gray,9,75,75) worse edge detection
# #lets see imgs side by side og and blur
# # openimg = cv.morphologyEx(bilateral,cv.MORPH_OPEN,kernel)
# # closeimg = cv.morphologyEx(gauss,cv.MORPH_CLOSE,kernel)
# edges = cv.Canny(gauss,75,200)
# #STEP 2############################
# # res = imutils.resize(edges,800)
# # cv.imshow("edge detected",res)
# # cv.waitKey(0)

# # edges = cv.dilate(edges, None, iterations=1)
# # edges = cv.erode(edges,None,iterations=1)

# # lines = cv.HoughLinesP(edges,1,np.pi/180,60,np.array([]),50,5)
# # for line in lines:
# #     for x1, y1, x2, y2 in line:
# #         cv.line(inp,(x1,y1),(x2,y2),(255,0,0),6)
# #         cv.line(edges,(x1,y1),(x2,y2),(255,0,0),3)

# # adjacentImg = np.concatenate((inp,edges),axis=1)

# # resizedImg = resizeWindow(adjacentImg)

# # cv.imshow('Edge Detection 1.0', resizedImg)
# # cv.imwrite('FailedEdgeDetection.png', resizedImg)
# # cv.imshow('compare img bilateral', resizedImg2)
# # cv.imshow("original", img)
# # cv.imshow("edges",edges)
# # cv.waitKey(0)
# cnts = cv.findContours(edges.copy(),cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# print("this is num of contours {}".format(len(cnts)))
# cnts = sorted(cnts, key = cv.contourArea,reverse=True)[:10]
# # loop over the contours
# for c in cnts:
# 	# approximate the contour
# 	peri = cv.arcLength(c, True)
# 	approx = cv.approxPolyDP(c, 0.02 * peri, True)
# 	# if our approximated contour has four points, then we
# 	# can assume that we have found our screen
# 	if len(approx) == 4:
# 		cardCnt = approx
# 		break
# # hull = cv.convexHull(cnts)
# #step 3 ###########################
# print("STEP 3. find contour corner points using polyDP ")
# cv.drawContours(edges,cardCnt,-1,(255,255,0),3)
# cv.drawContours(og,[cardCnt],-1,(0,255,0),3)
# adjacentImg = np.concatenate((gray2,edges),axis=1)
# resizedimg = resizeWindow(adjacentImg)
# cv.imshow("OG      4 point DP        edges",resizedimg)
# cv.waitKey(0)
# (cnt,_)= contours.sort_contours(cnt)
# ppM = None

# for c in cnt:

#     if cv.contourArea(c) < 12000:
#         continue

#     og = img.copy()
#     box = cv.minAreaRect(c)
#     box = cv.cv.boxPoints(box) if imutils.is_cv() else cv.boxPoints(box)
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
    # cv.waitKey(0)

# rng.seed(12345)
# def thresh_callback(val):
#     threshold = val

#     canny_output = cv.Canny(src_gray, threshold, threshold * 2)


#     contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


#     contours_poly = [None]*len(contours)
#     boundRect = [None]*len(contours)
#     centers = [None]*len(contours)
#     radius = [None]*len(contours)
#     for i, c in enumerate(contours):
#         contours_poly[i] = cv.approxPolyDP(c, 3, True)
#         boundRect[i] = cv.boundingRect(contours_poly[i])
#         centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])


#     drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)


#     for i in range(len(contours)):
#         color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
#         cv.drawContours(drawing, contours_poly, i, color)
#         cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
#           (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
#         cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)


#     cv.imshow('Contours', drawing)

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


#######################
"""""
 I'm getting the corners I need.
 Just need to test the distance
 really hope this works
 
 """


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="image path")
ap.add_argument("-f", "--file", help="file path")
args = ap.parse_args()
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv.imread(args.image)
ht = 800
ratio = image.shape[0] / ht
orig = image.copy()
image = imutils.resize(image, height=ht)
ppM = None
cc_width = 85.6  # unit in mm
cc_height = 53.98
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(gray, 75, 200)
# show the original image and the edge detected image
print("STEP 1: Edge Detection")
justEdge = np.nonzero(~edged)
cnts = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("this is num of contours {}".format(len(cnts)))
cnts = sorted(cnts, key=cv.contourArea, reverse=True) #43 is the height of cc in calTest6.
perie = cv.arcLength(cnts[43],True)
testapprox = cv.approxPolyDP(cnts[43],0.02*perie,True)
print("lenght of test approx is {}".format(len(testapprox)))
# cv.drawContours(image, [testapprox], -1, (0, 128, 255), 2)
# cv.imshow("image", image)
# cv.waitKey(0)
# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)
    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    # cv.drawContours(image, [c], -1, (0, 128, 255), 2)
    # cv.imshow("image", image)
    # cv.waitKey(0)
    if len(approx) == 4:
        screenCnt = testapprox
        break
    else:
        screenCnt = testapprox
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
print("this is num of contours {}".format(len(cnts)))

print(tuple(screenCnt[0][0])[0])
print(screenCnt)
print(screenCnt.shape)

tr_point =  tuple(screenCnt[0,0]) #tuple([613,160])
tl_point = tuple(screenCnt[1,0]) #tuple([477,161])
bl_point = tuple(screenCnt[2,0]) #tuple([476,246])
br_point = tuple(screenCnt[3,0]) #tuple([612,244])
point_5 = tuple(screenCnt[4,0])
point_6 = tuple(screenCnt[5,0])
# tltrX,tltrY = midpoint(tl_point,tr_point)
# blbrX,blbrY = midpoint(bl_point,br_point)
# tlblX,tlblY = midpoint(tl_point,bl_point)
# trbrX,trbrY = midpoint(tr_point,br_point)

# #channels are (B,G,R)
# cv.circle(image,(int(tltrX),int(tltrY)),13,(0,0,0),-1) #black
# cv.circle(image,(int(blbrX),int(blbrY)),13,(0,25,51),-1) #brown
# cv.circle(image,(int(tlblX),int(tlblY)),3,(255,0,127),-1) #purple
# cv.circle(image,(int(trbrX),int(trbrY)),3,(0,51,0),-1) #forest green
# print("these are the corner values of screencnt {}".format(screenCnt))
cv.circle(image,tr_point,6,(0,0,255),-1) #red bottom left with blue
# cv.circle(image,tl_point,3,(0,255,0),-1) #green bottom middle by p in presto
cv.circle(image,bl_point,3,(255,0,0),-1) #blue bottom left cal test 6
cv.circle(image,br_point,6,(0,255,255),-1) #yellow top left caltest 6
# cv.circle(image,point_5,6,(0,0,0),-1) #black top right by finger
cv.circle(image,point_6,6,(255,255,255),-1) #white top left with yellow


# cv.line(image,(int(tltrX),int(tltrY)),(int(blbrX),int(blbrY)),(255,0,255),2) #pink
# cv.line(image,(int(tlblX),int(tlblY)),(int(trbrX),int(trbrY)),(255,255,255),2) #white
dA = dist.euclidean(br_point,bl_point)
# dA = dist.euclidean((tltrX,tltrY),(blbrX,blbrY))
print("pixel height of cc {}".format(dA))
# meas = 
# # dA = dist.euclidean((tl_point[0],tl_point[1]),(bl_point[0],bl_point[1]))

# print("here is dA {}".format(dA))
# # dB = dist.euclidean((tlblX,tlblY),(trbrX,trbrY))
# # dB = dist.euclidean((bl_point[0],bl_point[1]),(br_point[0],br_point[1]))
# dB = dist.euclidean(bl_point,br_point)
# # print("here are coordinate for tlblX and tlblY {}".format((tlblX,tlblY)))
# # print("here are coordinate for trbrX and trbrY {}".format((trbrX,trbrY)))
# # print("here is db {}".format(dB))
if ppM is None:
    ppM = 3.01
    print("here is ppm {}".format(ppM))
# dimA = dA/ppM
# dimB = dB/ ppM

data = pd.read_csv(args.file)
of_data = getLandmark2D(data)
pix_dist = dist.euclidean(of_data['LICP'],of_data['RICP'])
print("this is pixel distance {}".format(pix_dist))
# licp, ricp, pdA = calcPD(data)
pdLine = cv.line(image,(int(of_data['LICP']['x']),int(of_data['LICP']['y'])),(int(of_data['RICP']['x']),int(of_data['RICP']['y'])),(255,255,255),2) #white
# print("convert series to tuple {}".format(tuple(licp)))
# #draw pupil markers on eye.
# licp[0] = int(licp[0])
# licp[1] = int(licp[1])
# print("LICP {}".format(licp[0]))
# cv.circle(image,tr_point,13,(0,0,255),-1) #red
# cv.circle(image,tl_point,3,(0,255,0),-1) #green
value = pix_dist/ppM
print("This is the PD measurement {}".format(value))
cv.drawContours(image, [screenCnt], -1, (0, 128, 255), 2)
cv.imshow("image", image)
cv.waitKey(0)
# cv.putText(image,"height{:.1f}mm".format(dimA),(int(tltrX -15),int(tltrY-10)),cv.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
# cv.putText(image,"width{:.1f}mm".format(dimB),(int(trbrX +1),int(trbrY)),cv.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
# print("Based on our PPI, this is the height of the credit card {}".format(dimA))
# cv.imshow("measurements", image)
# # cv.imwrite("img/CCedge.jpg",image)
# cv.waitKey(0)
# cv.destroyAllWindows()
