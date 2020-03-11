import numpy as np
import cv2
from time import sleep
from random import randint

cap = cv2.VideoCapture(0)

actions = ['action1', 'action2', 'action3', 'action4', 'action5', 'action6', 'actionAmplifier1', 'actionAmplifier2', 'actionAmplifier3', 'actionAmplifier4']

i = 0
for action in actions:
	while(True):
		i += 1
		# Capture frame-by-frame
		ret, frame = cap.read()
		# do what you want with frame
		#  and then save to file
		cv2.imwrite('./image2labellize/' + action + '/' + str(i) + '_' + str(randint(0, 1000)) + '_opencv.png', frame)

		cv2.imshow(action, frame.copy())

		keypress = cv2.waitKey(1) & 0xFF
		if keypress == ord("q"):
			cv2.destroyAllWindows()
			break
	sleep(1)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
