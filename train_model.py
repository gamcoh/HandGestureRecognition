import pandas as pd
import numpy as np
import cv2

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory('images/train', target_size=(135, 180), batch_size=32, class_mode='categorical', color_mode='rgb', shuffle=True, seed=0)
val_generator = train_datagen.flow_from_directory('images/val', target_size=(135, 180), batch_size=32, class_mode='categorical', color_mode='rgb', shuffle=True, seed=0)
test_generator = train_datagen.flow_from_directory('images/test', target_size=(135, 180), batch_size=32, class_mode='categorical', color_mode='rgb', shuffle=True, seed=0)

callbacks = [
	keras.callbacks.ModelCheckpoint('checkpoints', monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1),
	keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='batch')
]

model = Sequential()
model.add(Conv2D(18, (3, 3), activation='relu', input_shape=(135, 180, 3))) 
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(.2))
model.add(Conv2D(32, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(.2))
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(.2))
model.add(Conv2D(32, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(274, activation='relu'))
model.add(Dropout(.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=2000, epochs=10, verbose=1, validation_data=val_generator, validation_steps=800, callbacks=callbacks)

model.save('handrecognition_model.h5')

model = keras.models.load_model('handrecognition_model.h5')

test_loss, test_acc = model.evaluate_generator(test_generator)

print('Test accuracy: {:2.2f}%'.format(test_acc*100))

import matplotlib.pyplot as plt

from glob import glob
imagestest = glob('image_test/*')
imagestest = [ cv2.resize(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB), (180, 135)) for x in imagestest ]
imagestest = np.array(imagestest, dtype='float32')
imagestest = imagestest / 255
for image in imagestest:
	pred = model.predict(image.reshape(1, 135, 180, 3))
	plt.figure()
	plt.imshow(image)
	plt.text(10, 10, f"pred: {np.argmax(pred)}", bbox=dict(fill=True, edgecolor='red', linewidth=2))
	plt.show()

# keras load images !!
imagestest2 = []
for image in glob('image_test/*'):
	img = keras.preprocessing.image.load_img(image, grayscale=False, color_mode='rgb', target_size=(135, 180))
	img = keras.preprocessing.image.img_to_array(img)
	imagestest2.append(img)

imagestest2 = np.array(imagestest2)
imagestest2 = imagestest2 / 255
for image in imagestest2:
	pred = model.predict(image.reshape(1, 135, 180, 3))
	plt.figure()
	plt.imshow(image)
	plt.text(10, 10, f"pred: {np.argmax(pred)}", bbox=dict(fill=True, edgecolor='red', linewidth=2))
	plt.show()


imagetest_generator = train_datagen.flow_from_directory('image_test', target_size=(135, 180), batch_size=32, class_mode='categorical', color_mode='rgb', shuffle=True, seed=0)
i = 0
for batches in imagetest_generator:
	for y in range(len(batches[0])):
		image = batches[0][y]
		label = batches[1][y]
		pred = model.predict(image.reshape(1, 135, 180, 3))
		color = 'green' if np.argmax(label) == np.argmax(pred) else 'red'
		plt.figure()
		plt.imshow(image)
		plt.text(10, 10, f"label: {np.argmax(label)}", bbox=dict(fill=True, edgecolor=color, linewidth=2))
		plt.text(10, 25, f"pred: {np.argmax(pred)}", bbox=dict(fill=True, edgecolor=color, linewidth=2))
		plt.show()
	break
	if i == 10:
		break
	i += 1

print('TESTTT')
images_tests = []

pred = model.predict(X_train[:10])

image_test = cv2.imread('./image_test/2020-02-16-002747.jpg')
img = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (180, 135)) # Reduce image size so training can be faster
X = np.array(img, dtype="float32")
X = X.reshape(1, 135, 180, 3) # Needed to reshape so CNN knows it's different images
X = X / 255
print(np.argmax(pred), labels[np.argmax(pred)])

