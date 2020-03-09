from tensorflow.keras.models import load_model
import cv2
import numpy as np

class Prediction():
	def __init__(self):
		self.model = load_model('./handrecognition_model.h5')

	def get_action(self, image):
		img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (180, 135))
		x = np.array(img, dtype="float32")
		x = x.reshape(1, 135, 180, 1)
		pred = self.model.predict(x)
		
		labels = {
			"action1": 0, # thumb touching index
			"action2": 1, # index up
			"action3": 2, # index down
			"action4": 3, # thumb up
			"action5": 4, # fist
			"action6": 5, # thumb down
			"actionAmplifier1": 6, # 2 fingers
			"actionAmplifier2": 7, # 3 fingers
			"actionAmplifier3": 8, # 4 fingers
			"actionAmplifier4": 9 # 5 fingers
		}
		labels = {value:key for key, value in labels.items()}
		return labels[np.argmax(pred)]

