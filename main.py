import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import optimizers

from tensorflow.keras.applications import VGG16

imagepaths = []

# Go through all the files and subdirectories inside a folder and save path to images inside list
for root, dirs, files in os.walk("./image2labellize/.", topdown=False):
  for name in files:
    path = os.path.join(root, name)
    if path.endswith(("jpg", "jpeg", "png")): # We want only the images
      imagepaths.append(path)

X = []
y = []

# Loops through imagepaths to load images and labels into arrays
for path in imagepaths:
  img = cv2.imread(path) # Reads image and returns np.array
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converts into the corret colorspace (GRAY)
  img = cv2.resize(img, (320, 240)) # Reduce image size so training can be faster
  X.append(img)
  # Processing label in image path
  category = path.split("\\")[-2]
  y.append(category)

# Turn X and y into np.array to speed up train_test_split
X = np.array(X, dtype="float32")
X = X.reshape(len(imagepaths), 240, 320, 1) # Needed to reshape so CNN knows it's different images
y = np.array(y)

le = LabelEncoder()
y = le.fit_transform(y)

np.save('y.npy', y)
np.save('X.npy', X)
np.save('classes.npy', le.classes_)

# X = np.load('X.npy')
# y = np.load('y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# create the base model with imagenet weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])

# add a flattening layer
# let's add a fully-connected layer to classify
x = base_model.get_layer('block4_pool').output
x = Flatten(name='Flatten')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(X_test, y_test))

model.save('handrecognition_model.h5')

test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy: {:2.2f}%'.format(test_acc*100))

print('TESTTT')
image_test = cv2.imread('.\images\customgestures\\action1\\125_163_opencv.png')

img = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (320, 240))
X = np.array(img, dtype="float32")
X = X.reshape(1, 240, 320, 1)
pred = model.predict(X)
print(np.argmax(pred))
classes = [
  'acc1',
  'acc2',
  'acc3',
  'acc4',
  'acc5',
  'acc6',
  'accA1',
  'accA2',
  'accA3',
  'accA4',
]
print(np.argmax(pred), classes[np.argmax(pred)])
