import cv2.cv2 as cv
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

######################
wCam,hCam= 640,480
######################

cap = cv.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

pTime = 0

detector = htm.handDetector(detectionCon=0.7)
# From AndreMiras/pycaw @github
# pip install pycaw

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_ , CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume.GetVolumeRange()
# print(volume.GetVolumeRange())
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
#Thanks Andre
volBar=400
volPer = 0

while True:
    success, image = cap.read()
    image = detector.findHands(image)
    lmList = detector.findPosition(image, draw=False)
    # if len(lmList)!=0:
        # print(lmList[2])

    # 4----> Tip of thumb
    # 8---->Tip of the index This is needed for the gesture control
    if len(lmList)!=0:
        # print(lmList[4], lmList[8])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cv.circle(image, (x1,y1), 10, (0,255,0), cv.FILLED)
        cv.circle(image, (x2, y2), 10, (0, 255, 0), cv.FILLED)
        cv.line(image, (x1, y1), (x2, y2), (0,255,0),2 )
    #     Finding midpoints
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv.circle(image, (cx, cy), 10, (0, 255, 0), cv.FILLED)


    # Finding Length
        length = math.hypot(x2-x1,y2-y1)
        # print(length)

        # Hand Range  20 - 240
        # Volume Range -96 - 0
        vol = np.interp(length,[20,240],[minVol,maxVol])
        volBar = np.interp(length,[20,240],[400,150])
        volPer = np.interp(length, [20, 240], [0,100])
        print(int(length), vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<30:
            cv.circle(image, (cx, cy), 10, (100, 0, 255), cv.FILLED)


    cv.rectangle(image,(50,150), (85,400), (30,60,30), 3)
    cv.rectangle(image, (50, int(volBar)), (85, 400), (0, 255, 0), cv.FILLED)
    cv.putText(image, f'{int(volPer)}%', (40, 450), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(image,f'FPS:{int(fps)}', (30,70), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

    cv.imshow("Image",image)
    cv.waitKey(1)
