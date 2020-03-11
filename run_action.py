import cv2

from Actions import Actions
from Predictions import Prediction

cam = cv2.VideoCapture(0)

model = Prediction() # init the trained model for inference
action = Actions() # init the action class in order to lunch actions based on predictions

while True:
	ret, frame = cam.read()
	
	if not ret:
		break

	pred = model.get_action(frame)
	cv2.imshow('hand', frame.copy())

	action.run(pred)

	keypress = cv2.waitKey(1) & 0xFF
	if keypress == ord("q"):
		cv2.destroyAllWindows()
		break

# When everything's done, release the capture
cam.release()
cv2.destroyAllWindows()