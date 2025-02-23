import numpy as np
import cv2

def order_points(pts):
    rect = np.zeros((4,2),dtype='float32')
    # ordering is top left > top right > bottom right > bottom left
    #I will prolly switch up the order. in which case diff will be setup differently
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_pt_transform(image,pts):

    rect = order_points(pts)
    (tl,tr,br,bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
	    [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
# compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
# return the warped image
    return warped

# image = cv2.imread('calibrategum.jpg')
# pts = np.array([(501,185),(893,151),(979,323),(687,441)], dtype = "float32")

# # the image
# warped = four_pt_transform(image, pts)
# # show the original and warped images
# cv2.imshow("Original", image)
# cv2.imshow("Warped", warped)
# cv2.waitKey(0)