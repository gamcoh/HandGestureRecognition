import pandas as pd
import numpy as np

from utils import Utils

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import models
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

train_generator, val_generator, test_generator = Utils.get_generators(target_size=(135, 180), batch_size=32)

vgg16 = VGG16(include_top=False, weights="imagenet", input_shape=(135, 180, 3))

for layer in vgg16.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(vgg16.output)
x = Dense(256, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

model = Model(inputs=vgg16.input, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = [
    ModelCheckpoint('vgg16_freeze_256_img_135.h5', save_best_only=True),
    EarlyStopping(patience=5)
]

model.fit_generator(
    train_generator,
    steps_per_epoch=200,
    epochs=20,
    validation_data=val_generator,
    validation_steps=200,
    callbacks=callbacks
)
