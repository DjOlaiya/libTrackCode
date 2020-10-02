import cv2
# vidcap = cv2.VideoCapture('./webcamChessboard/calibrationvid.mp4')
# def getFrame(sec):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#     success,image = vidcap.read()
#     if success:
#         cv2.imwrite("./webcamChessboard/image{}.jpg".format(count), image)     # save frame as JPG file
#         print("saved frame {}".format(count))
#     return success
# sec = 0
# frameRate = 0.5 #//it will capture image in each 0.5 second
# count=1
# success = getFrame(sec)
# while success:
#     count = count + 1
#     sec = sec + frameRate
#     sec = round(sec, 2)
#     success = getFrame(sec)


# import cv2
vidcap = cv2.VideoCapture('./webcamChessboard/calibrationvid.mp4')
success, image = vidcap.read()
count = 1
total = 1018
while (count < total):
  cv2.imwrite("./webcamChessboard/image{}.jpg".format(count), image)    
  success, image = vidcap.read()
  print('Saved image ', count)
  count += 8

# cap = cv2.VideoCapture('./webcamChessboard/calibrationVid.mp4')

# while(cap.isOpened()):
#     ret, frame = cap.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()