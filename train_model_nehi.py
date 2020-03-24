from utils import Utils
from tensorflow import keras

TARGET_SIZE = (135, 180, 3)
train_generator, val_generator, test_generator, imagetest_generator = Utils.get_generators(target_size=TARGET_SIZE[:2], batch_size=252)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(18, (3, 3), activation='relu', input_shape=TARGET_SIZE)) 
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(18, (3, 3), activation='relu')) 
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu')) 
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu')) 
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(.3))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model = keras.models.load_model('nehi.h5')
lr_schedule = keras.callbacks.LearningRateScheduler(Utils.scheduler, verbose=1)

checkpoint = keras.callbacks.ModelCheckpoint("nehi_2th_gen.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30, verbose=1, mode='auto')
model.fit_generator(train_generator, steps_per_epoch=300, epochs=500, verbose=1, validation_data=val_generator, validation_steps=150, callbacks=[early, checkpoint, lr_schedule])
