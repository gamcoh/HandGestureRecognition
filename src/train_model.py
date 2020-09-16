import sys
sys.path.append('../utils')

from Utils import Utils

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

WIDTH = 202
HEIGHT = 280
BATCH = 32

train_generator, val_generator, test_generator = Utils.get_generators(target_size=(WIDTH, HEIGHT), batch_size=BATCH)

vgg16 = VGG16(include_top=False, weights="imagenet", input_shape=(WIDTH, HEIGHT, 3))

for layer in vgg16.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(vgg16.output)
x = Dense(254, activation='relu')(x)
x = Dropout(.5)(x)
x = Dense(254, activation='relu')(x)
x = Dropout(.5)(x)
x = Dense(11, activation='softmax')(x)

model = Model(inputs=vgg16.input, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = [
    #TensorBoard(log_dir='../logs', histogram_freq=0, write_images=True, update_freq='batch'),
    ModelCheckpoint('vgg16_freeze_254x2_img_202_b32.h5', save_best_only=True),
    EarlyStopping(patience=5)
]

def check_gen(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass

model.fit(
    check_gen(train_generator),
    epochs=20,
    steps_per_epoch=40605//BATCH,
    validation_steps=300,
    validation_data=check_gen(val_generator),
    callbacks=callbacks
)
