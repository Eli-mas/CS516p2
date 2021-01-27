"""this script furnishes one function, 'crate_model',
for creating a Convolutional Neural Network model."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import optimizers

from cnn_params import *

def create_model(x_train, y_train, pooler = MaxPooling2D):
	print('constructing model... ', end='', flush=True)
	model = Sequential()
	model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (5, 5)))
	model.add(Activation('relu'))
	if POOL:
		model.add(pooler(pool_size=(3, 3)))
	else:
		model.add(Conv2D(64, (3, 3), strides = (2, 2)))
		model.add(Activation('relu'))
	model.add(SpatialDropout2D(0.25))
	
	model.add(Conv2D(128, (5, 5), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(128, (7, 7)))
	model.add(Activation('relu'))
	if POOL:
		model.add(pooler(pool_size=(2, 2)))
	else:
		model.add(convolver2(window=(8, 8), strides = (7, 7)))
		model.add(Activation('relu'))
	model.add(SpatialDropout2D(0.25))
	
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(y_train.shape[1]))
	model.add(Activation(final_activator))
	
	# RMSprop optimizer
	opt = optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
	
	# train the model using RMSprop
	model.compile(loss = loss,
				  optimizer=opt,
				  metrics=['accuracy'])
	print('done!', flush=True)
	print()
	model.summary()
	print()
	return model
