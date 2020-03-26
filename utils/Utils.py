from tensorflow import keras

def get_generators(target_size: tuple = (135, 180), batch_size: int = 32) -> tuple:
	train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
	train_generator = train_datagen.flow_from_directory('images/train', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)
	val_generator = train_datagen.flow_from_directory('images/val', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)
	test_generator = train_datagen.flow_from_directory('images/test', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)
	imagetest_generator = train_datagen.flow_from_directory('image_test', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)
	return train_generator, val_generator, test_generator, imagetest_generator

def scheduler(epoch):
    if epoch < 200:
        return .001
    if epoch < 400:
        return .0005

    return .0001
