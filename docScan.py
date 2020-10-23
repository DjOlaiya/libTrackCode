from four_pt_transform import four_pt_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils


def save_toFile(edge, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("EdgeMatrix", edge)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", help="image path")
args = ap.parse_args()
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(args.image)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
# show the original image and the edge detected image
print("STEP 1: Edge Detection")
# cv2.imshow("Image", image)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
justEdge = np.nonzero(~edged) 
"""
Ok this is important. the ~ operator reverses the bits of an obj.
The formula is essentially: (-x) - 1 for any x. So how does that help here?
 """
print(type(edged))
save_toFile(np.asarray(justEdge), 'edgeArray.yml')
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("this is num of contours {}".format(len(cnts)))
cnts = sorted(cnts, key = cv2.contourArea,reverse=True)[:10]
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
print(tuple(screenCnt[2,0]))
print(tuple(screenCnt))
tr_point =  tuple(screenCnt[0,0]) #tuple([613,160])
tl_point = tuple(screenCnt[1,0]) #tuple([477,161])
bl_point = tuple(screenCnt[2,0]) #tuple([476,246])
br_point = tuple(screenCnt[3,0]) #tuple([612,244])
print("these are the corner values of screencnt {}".format(screenCnt))
cv2.circle(image,tr_point,4,(0,0,255),-1) #red
cv2.circle(image,tl_point,4,(0,255,0),-1) #green
cv2.circle(image,bl_point,4,(255,0,0),-1) #blue
cv2.circle(image,br_point,8,(0,255,255),-1) #yellow
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_pt_transform(orig, screenCnt.reshape(4, 2) * ratio)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
# show the original and scanned images
print("STEP 3: Apply perspective transform")
# cv2.imshow("Original", imutils.resize(orig, height = 650))
# cv2.imshow("Scanned", imutils.resize(warped, height = 650))
# cv2.waitKey(0)

