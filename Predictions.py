from tensorflow.keras.models import load_model
import cv2
import numpy as np

class Prediction():
	def __init__(self):
		self.model = load_model('./handrecognition_model.h5')

	def get_action(self, image):		
		img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, (320, 240))
		x = np.array(img, dtype="float32")
		x = x.reshape(1, 240, 320, 1)
		pred = self.model.predict(x)
		classes = np.load('./classes.npy')

		return classes[np.argmax(pred)]

