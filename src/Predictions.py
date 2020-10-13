from tensorflow.keras.models import load_model
import cv2
import numpy as np

class Prediction():
    def __init__(self):
        self.model = load_model('../models/vgg16_freeze_142x2_img_320_b86.h5')
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
            "other"
        ]


    def get_action(self, image):
        """Predict the action based on the imahe

        Arguments:
            image {numpy.ndarray} -- an image

        Returns:
            string -- the predicted action label
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 240))
        x = np.array(img, dtype="float32")
        x = x.reshape(1, 240, 320, 3)
        pred = self.model.predict(x)

        return self.labels[np.argmax(pred)]

