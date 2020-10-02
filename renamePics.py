import cv2
import glob
import os


images = glob.glob('./newChessboard/*.jpg')
print(len(images))
count = 0
for fname in images:
    try:
        os.rename(fname, "./newChessboard/nuChessboard{}.jpg".format(count))
        count+=1
    except:
        print(f"could not rename file")

print("rename complete")