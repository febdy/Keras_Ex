import numpy as np  # matrix operations (ie. difference between two matricies)
import cv2  # (OpenCV) computer vision functions (ie. tracking)

ERODE = True

fgbg = cv2.createBackgroundSubtractorMOG2()

video = cv2.VideoCapture('C:/Users/BIT-USER/Desktop/python_workplace/HUN.mp4')

while True:
    success, frame = video.read()

    if success:
        fgmask = fgbg.apply(frame)

        if ERODE:
            fgmask = cv2.erode(fgmask, np.ones((3, 3), dtype=np.uint8), iterations=1)

        cv2.imshow('frame', fgmask)

        if cv2.waitKey(1) & 0xff == 27:
            break
    else:
        break

cv2.destroyAllWindows()
video.release()
