from random import randint
from time import sleep

import cv2

ACTIONS = [
    'action1',
    'action2',
    'action3',
    'action4',
    'action5',
    'action6',
    'actionAmplifier1',
    'actionAmplifier2',
    'actionAmplifier3',
    'actionAmplifier4',
    'other'
]

cap = cv2.VideoCapture(0)

i = 0
for action in ACTIONS:
    while True:
        i += 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        # save the image to a file in the right folder
        FILENAME = './image2labellize/'+action+'/'+str(i)+'_'+str(randint(0, 1000))+'_opencv.png'
        cv2.imwrite(FILENAME, frame)

        img = frame.copy()
        img = cv2.flip(img, 1)
        cv2.putText(img, action, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('reflection', img)

        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            cv2.destroyAllWindows()
            break
    sleep(2)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
