import cv2

from utils import Utils
from time import sleep, time
from Actions import Actions
from Predictions import Prediction

model = Prediction() # init the trained model for inference
action = Actions() # init the action class in order to lunch actions based on predictions

while True:
	sleep(30) # search for predictions every 30 seconds

	cam = cv2.VideoCapture(0)
	preds = []
	for frame in Utils.getFrames(cam, s=5):
		preds.append(model.get_action(frame))
	cam.release()

	# find the 2 most common preds in all the preds made
	# once we got them we want to see if there is an amplifier 
	# if yes we execute both actions: amplifier, actionX
	# if not we only search for the one most action found and execute it
	most_common_preds = Utils.top_k(preds, k=2)
	amplifierFounded, actionsWanted = Utils.hasAmplifier(most_common_preds)

	if amplifierFounded:
		for acc in actionsWanted:
			action.run(acc)
	else:
		action.run(Utils.top_k(preds, k=1)[0])

	keypress = cv2.waitKey(1) & 0xFF
	if keypress == ord("q"):
		cv2.destroyAllWindows()
		break

# When everything's done, release the capture
cv2.destroyAllWindows()
