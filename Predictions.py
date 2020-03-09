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
			"action1": 0,
			"action2": 1,
			"action3": 2,
			"action4": 3,
			"action5": 4,
			"action6": 5,
			"actionAmplifier1": 6,
			"actionAmplifier2": 7,
			"actionAmplifier3": 8,
			"actionAmplifier4": 9
		}
		labels = {value:key for key, value in labels.items()}
		return labels[np.argmax(pred)]

