"""The main() function in this module trains, saves, and evaluates
a neural network. It also automatically provides email updates on the
progress of the script."""
import os, sys, subprocess, datetime, traceback

"""
>>> nohup python cnn.py & disown
this command allows it to run in the background, even when logged out of ssh
"""

print('\n\n')
for _ in range(3): print('\t'.join("*" for _ in range(8)))


import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from cnn_util import *
from p2_entries import *
from cnn_model import create_model

def train_and_save_network(attr=None):
	if attr is None: attr=sys.argv[1]
	assert (attr is not None)
	subject = f'cnn.py {attr} {datetime.datetime.now()}'
	subprocess.run('> nohup.out', shell=True)
	os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
	
	GPU = get_available_device()
	print('*   *   *   device to be used:%s   *   *   *'%GPU)
	
	
	globals().update(get_data(attr=attr))
	
	model = create_model(x_train, y_train)
	
	callback = CallbackMailer(subject)
	if not data_augmentation:
		print('Not using data augmentation.')
		with tf.device(GPU):
			hist = model.fit(x_train, y_train,
					  batch_size=batch_size,
					  epochs=epochs,
					  validation_data=(x_valid, y_valid),
					  shuffle=True,
					  callbacks = [callback])
	else:
		print('Using real-time data augmentation.')
		# preprocessing and realtime data augmentation
		datagen = ImageDataGenerator(
			featurewise_center=False,  # set input mean to 0 over the dataset
			samplewise_center=False,  # set each sample mean to 0
			featurewise_std_normalization=False,  # divide inputs by std of the dataset
			samplewise_std_normalization=False,  # divide each input by its std
			zca_whitening=False,  # apply ZCA whitening
			zca_epsilon=1e-06,  # epsilon for ZCA whitening
			rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
			# randomly shift images horizontally (fraction of total width)
			width_shift_range=0.1,
			# randomly shift images vertically (fraction of total height)
			height_shift_range=0.1,
			shear_range=0.,  # set range for random shear
			zoom_range=0.,  # set range for random zoom
			channel_shift_range=0.,  # set range for random channel shifts
			# set mode for filling points outside the input boundaries
			fill_mode='nearest',
			cval=0.,  # value used for fill_mode = "constant"
			horizontal_flip=True,  # randomly flip images
			vertical_flip=False,  # randomly flip images
			# set rescaling factor (applied before any other transformation)
			rescale=None,
			# set function that will be applied on each input
			preprocessing_function=None,
			# image data format, either "channels_first" or "channels_last"
			data_format=None,
			# fraction of images reserved for validation (strictly between 0 and 1)
			validation_split=0.0)
	
		# Compute quantities required for feature-wise normalization
		# (std, mean, and principal components if ZCA whitening is applied).
		datagen.fit(x_train)
	
		with tf.device(GPU):
			# Fit the model on the batches generated by datagen.flow().
			hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
								epochs=epochs,
								validation_data=(x_valid, y_valid))
	
	# Save model and weights
	print('saving model... ',end='',flush=True)
	model_path = raw_model_path.format(attr=attr)
	model.save(model_path)
	print('done! >>> Saved trained model at {}'.format(model_path))
	
	write_model_run(hist, model, attr, callback, subject)
	
	# Score trained model.
	scores = np.array(model.evaluate(x_test, y_test, verbose=1))#df.loc['test'].values
	print('Test loss:', scores[0])
	print('Test accuracy:', scores[1])
	
	mail(
		subject = subject,
		message = "cnn.py is finished running: loss, accuracy = {}".format(scores.tolist())
	)

def main():
	print('running cnn.py')
	try: train_and_save_network()
	except Exception as e:
		print('an exception was encoutered: <{}>'.format(repr(e)))
		traceback.print_exc()
		exc_mes = "exception occured: <{}>: check nohup.out for trace if applicable".format(repr(e))
		mail(
			subject=subject,
			message = exc_mes
		)
		subprocess.run(
			'echo "{mes}" | mail -s "{sub} error" {a}'.format(mes=exc_mes,sub=subject,a=EMAIL),
			shell=True)
		#raise

if __name__=='__main__':
	main()