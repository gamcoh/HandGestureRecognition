import pandas as pd
import numpy as np

from utils import Utils
from tensorflow import keras

train_generator, val_generator, test_generator, imagetest_generator = Utils.get_generators(target_size=(135, 180), batch_size=32)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(18, (3, 3), activation='relu', input_shape=(135, 180, 3))) 
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(18, (3, 3), activation='relu')) 
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu')) 
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu')) 
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(.2))
model.add(keras.layers.Dense(10, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint = keras.callbacks.ModelCheckpoint("vgg16_5.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30, verbose=1, mode='auto')
model.fit_generator(train_generator, steps_per_epoch=100, epochs=200, verbose=1, validation_data=val_generator, validation_steps=10, callbacks=[early, checkpoint])

model.save('handrecognition_model_040.h5')

test_loss, test_acc = model.evaluate_generator(test_generator)

print('Test accuracy: {:2.2f}%'.format(test_acc*100))

import matplotlib.pyplot as plt

test_loss, test_acc = model.evaluate_generator(imagetest_generator)
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

