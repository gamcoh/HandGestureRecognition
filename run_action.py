import numpy as np
from time import sleep
import cv2
import os

from Actions import Actions
from Predictions import Prediction

cam = cv2.VideoCapture(0)

model = Prediction()
action = Actions()

while True:
	ret, frame = cam.read()

	frame = cv2.imread('./images_test/2020-03-09-171411.jpg')

	pred = model.get_action(frame)
	cv2.imshow('hand', frame.copy())

	print(pred)
	# os.system(f'notify-send "pred: {pred}"')
	# action.run(pred)

	keypress = cv2.waitKey(1) & 0xFF
	if keypress == ord("q"):
		cv2.destroyAllWindows()
		break
	break

# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()