# Minimum code to execute a Hand tracking procedure
import time

import cv2.cv2 as cv
import mediapipe as mp

# Creating our video object
cap = cv.VideoCapture(0)
# To create a module to get point 5 of the hand
mpHands = mp.solutions.hands
hands = mpHands.Hands()
'''
                                ----Hands()----
def __init__(self,
               static_image_mode=False,-->so that when it detects if it has a good tracking confidence it will keep track
               max_num_hands=2,
               model_complexity=1,
               min_detection_confidence=0.5,---->50%
               min_tracking_confidence=0.5) ----> if it goes below 50% it will go back to detection
               
              * it will track when it has a good tracking confidence else it will continue detecting
'''
mpDraw = mp.solutions.drawing_utils
previousTime = 0
currentTime = 0

while True:
    success, image = cap.read()
    # send in our rgb image to the hands object as the object only uses RGB images
    imgRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    ''' Gives coordinates of multiple hands
    landmark {
  x: 0.40811359882354736
  y: 0.3804309070110321
  z: 0.009026714600622654
}
    '''

    # We have to extract the hand results one by one
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            # Getting info from within these hands
            # Each id has a corresponding landmark and landmark has an x, y and z
            for idNumber, landmarkInformation in enumerate(handLandmarks.landmark):
                # print(idNumber, landmarkInformation)
                h, w, c = image.shape
                #     height,width and channel of the image
                cx, cy = int(landmarkInformation.x * w), int(landmarkInformation.y * h)
                print(idNumber, cx, cy)
                if idNumber == 0:
                    cv.circle(image, (cx, cy), 25, (255, 0, 255), cv.FILLED)
                if idNumber == 4:
                    cv.circle(image, (cx, cy), 15, (255, 0, 255), cv.FILLED)
            #         circle(img, center, radius, color, thickness=None, lineType=None, shift=None)
            # We can thus track a part of our hand like this

            mpDraw.draw_landmarks(image, handLandmarks, mpHands.HAND_CONNECTIONS)
    # Here if there are multiple hands detected then enter the for loop
    # for every hand detected perform the following drawing operations
    # using draw_landmarks() draw on the img not imgRGB as we are displaying only thr BGR image as output not the RGB one
    # mpHands.HAND_CONNECTIONS for drawing connections between dots

    # Displaying the Frames per sec
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime
    # We're extracting the current time, i.e the new time and we find the fps
    # We update the previous time with the current time
    cv.putText(image, str(int(fps)), (10, 70), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 3)
    # putText(image, fps(string), position, font, scale, color, thickness

    cv.imshow("Image", image)
    cv.waitKey(1)

# Now we have to create a module out of this program so that we don't have to repeat code again and again
# We can simply as for the list of these values like say 10(idNumber)