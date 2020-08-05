from tensorflow import keras

def test_model(target_size: tuple, batch_size: int = 64, model_name: str = 'nehi') -> None:
	train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
	test_generator = train_datagen.flow_from_directory('images/test', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)
	imagetest_generator = train_datagen.flow_from_directory('image_test', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)

	model = keras.models.load_model(f'{model_name}.h5')

	print('=*' * 100)
	print(' === {} ==='.format(model_name.upper()))
	print(model.evaluate_generator(imagetest_generator))
	print(model.evaluate_generator(test_generator))
	

test_model(target_size=(135, 180), batch_size=252, model_name='nehi_3th_gen')
