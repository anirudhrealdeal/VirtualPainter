import cv2.cv2 as cv
import time
import mediapipe as mp
import numpy as np
import os
import HandTrackingModule as htm

#####################
brushThickness = 15
eraserThickness = 50
#####################
folderPath = "Header"
myList = os.listdir(folderPath)
# print(myList)
overlayList =[]
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))
header = overlayList[0]

drawColor = (255,100,255)


cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 1280)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720,1280,3), np.uint8)

while True:
    # Step1: Import the image
    success, img = cap.read()
    # We have to flip the image horizontally
    img = cv.flip(img, 1)
    # Step 2: Find the Hand landmarks(using hand tracking module)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList)!= 0:
        # print(lmList)
        x1, y1 = lmList[8][1:] # Landmark of the tip of the index finger(0 is just the indexes for example 8 but 1 and 2 is x and y coord)
        x2, y2 = lmList[12][1:] # Landmark of the tip of the middle finger


        # Step 3: Check which fingers are up(draw when 1 finger is up and select when 2 fingers are up)
        fingers = detector.fingeraUp()
        # print(fingers)
        # Step 4: If two fingers are up ie selection mode
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv.rectangle(img, (x1, y1-50), (x2, y2+50), drawColor, cv.FILLED)
            # print(x2)
            # print("Selection Mode")
            # Checking for the click
            if y1< 221:
                if 250 < x1 < 450:
                    header= overlayList[0]
                    drawColor = (255, 100, 255)
                elif 500< x1 <617:
                    header = overlayList[1]
                    drawColor = (0,190,0)
                elif 730< x1 <840:
                    header = overlayList[2]
                    drawColor = (0,165,255)
                elif 960 < x1 < 1200:
                    header = overlayList[3]
                    drawColor= (0,0,0)


        # Step 5: If index finger is up ie drawing mode
        if fingers[1] and fingers[2]==False:
            cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
            # print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp =x1, y1

            if drawColor == (0,0,0):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)


            cv.line(img, (xp,yp), (x1, y1), drawColor, brushThickness)
            cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1,y1
    # We are converting to binary image and inversing it
    imgGray = cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)
    # We are creating inverse image so that all that we drew would come in black
    _, imgInv = cv.threshold(imgGray, 50,255,cv.THRESH_BINARY_INV)
    # We are converting back as we want to add it to our original image
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img, imgInv)
    img = cv.bitwise_or(img,imgCanvas)



    # Setting the header image
    img[0:221, 0:1280] = header
    # img = cv.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv.imshow("Image", img)
    cv.imshow("Image2", imgCanvas)
    cv.imshow("Image3", imgInv)
    cv.waitKey(1)
