import time
import cv2.cv2 as cv
import mediapipe as mp
import HandTrackingModule as htm

previousTime = 0
    currentTime = 0
    cap = cv.VideoCapture(0)
    detector = htm.handDetector()

    while True:
        success, image = cap.read()
        image = detector.findHands(image)
        lmList = detector.findPosition(image)
        # if we dont want to see the colour and drawing and just want to get the position of the thumb
        # image = detector.findHands(image, draw=False)
        # And if we dont want the custom draw again initialize the draw to false
        # lmList = detector.findPosition(image)

        if len(lmList)!=0:
            print(lmList[4])

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        cv.putText(image, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
        cv.imshow("Image", image)
        cv.waitKey(1)