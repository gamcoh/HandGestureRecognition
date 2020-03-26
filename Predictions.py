from tensorflow.keras.models import load_model
import cv2
import numpy as np

class Prediction():
	def __init__(self):
		self.model = load_model('./nehi_2th_gen.h5')
		self.labels = [
			"action1",
			"action2",
			"action3",
			"action4",
			"action5",
			"action6",
			"actionAmplifier1",
			"actionAmplifier2",
			"actionAmplifier3",
			"actionAmplifier4",
		]


	def get_action(self, image):
		"""Predict the action based on the imahe
		
		Arguments:
			image {numpy.ndarray} -- an image
		
		Returns:
			string -- the predicted action label
		"""		
		img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (180, 135))
		x = np.array(img, dtype="float32")
		x = x.reshape(1, 135, 180, 3)
		pred = self.model.predict(x)
		
		print(pred, np.argmax(pred), self.labels[np.argmax(pred)])

		return self.labels[np.argmax(pred)]

