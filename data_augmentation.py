from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from glob import glob

def aug(filename):
	# load the image
	img = load_img(filename)
	# convert to numpy array
	data = img_to_array(img)
	# expand dimension to one sample
	image = expand_dims(data, 0)
	# create image data augmentation generator
	datagen = ImageDataGenerator(rotation_range=50, width_shift_range=0.1, height_shift_range=0.1,shear_range=0.15, zoom_range=0.1,channel_shift_range=10, horizontal_flip=True)
	# prepare iterator
	datagen.fit(image)
	it = datagen.flow(image, save_to_dir='./images/customgestures/actionAmplifier4', save_format='jpeg', save_prefix='aug_')

	for x, val in zip(it, range(10)):
		pass

for filename in glob('./images/customgestures/actionAmplifier4/*.jpg'):
	aug(filename)
