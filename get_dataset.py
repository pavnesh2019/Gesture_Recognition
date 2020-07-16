# THIS FILE IS CREATED FOR QUICKLY GENERATING DATASET

### importing modules
import cv2
import imutils
import numpy as np
from time import time

initTime = time()



background = None

## This function finds the average over background
def avg_bg(img, accumWeight):
    
    global background
    if background is None:
        background = img.copy().astype("float")
        return
    cv2.accumulateWeighted(img, background, accumWeight)

## This function does hand segmentation
def segment_hand(img):
    global background
    threshold = 30
    diff = cv2.absdiff(background.astype("uint8"), img)

    thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresh, segmented)


## main loop
if __name__ == "__main__":
    accumWeight = 0.5


    cap = cv2.VideoCapture(0)


    top, right, bottom, left = 70, 350, 270, 550

    num_frames = 0
    imageNumber = 0
    calibrated = False

    while True:
        ret, frame = cap.read()


        if ret == True:
            
            frame = imutils.resize(frame, width=700)
            frame = cv2.flip(frame, 1)
            clone = frame.copy()

            height, width = frame.shape[:2]
            
            roi = frame[top:bottom, right:left]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if num_frames < 30:
                avg_bg(gray, accumWeight)
                if num_frames == 1:
                    print("[INFO] Callibrating....")
                if num_frames == 29:
                    print("[INFO] Done Callibrating....")
            else:
                hand = segment_hand(gray)
                if hand is not None:
                    (thresholded, segmented) = hand
                    
                    # show the thresholded image
                    cv2.imshow("Thesholded", thresholded)
                    # mannual callibration
                    if time() - initTime > 10:
                    # Set the directory CORRECTLY
                        directory = "C:/Users/Pavnesh Chaturvedi/Documents/Gesture_Rcognition/dataset/hand4("+str(imageNumber)+").jpg"
                        cv2.imwrite(directory,thresholded)
                        imageNumber += 1
                        print(directory)
            
            cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

            num_frames += 1
            
            cv2.imshow("Video Feed", clone)

            keypress = cv2.waitKey(1) & 0xFF

            if keypress == ord("q"):
                break


        else:
            print("[INFO] No Frame....")
            
cap.release()
cv2.destroyAllWindows()