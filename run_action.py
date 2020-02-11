import numpy as np
from time import sleep
import cv2
import os

from Actions import Actions
from Predictions import Prediction

cap = cv2.VideoCapture(0)

model = Prediction()
action = Actions()

i = 0
while True:
	sleep(2)
	i += 1
	ret, frame = cap.read()

	# test
	# if i == 1:
	# 	frame = cv2.imread('./images/customgestures/actionAmplifier2/2020-02-04-131407.jpg')
	# else:
	# frame = cv2.imread('./images/customgestures/action1/2020-02-04-131204.jpg')

	pred = model.get_action(frame)

	action.run(pred)

	keypress = cv2.waitKey(1) & 0xFF
	if keypress == ord("q"):
		cv2.destroyAllWindows()
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()