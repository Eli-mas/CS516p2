# Find misclassified images.

import os, sys
from cnn_util import *
from sklearn.metrics import confusion_matrix
#------------------------------------------------------------
# load the model.

import tensorflow as tf


#---------------------------------------------------------------------------------
# read the labels, and split them between training, validation, and test sets.

import numpy as np

#---------------------------------------
# make the predictions.

def get_misclassifications(attr):
	return __get_misclassifications(attr)

def __get_misclassifications(attr):
	attr = get_run_attribute(attr)
	try:
		y_pred = np.load(f'y_pred/y_pred_{attr}.npy')
		data = get_data(attr=attr,short=True)
		y_test = data['y_test']
		y_train = data['y_train']
		y_valid = data['y_valid']
	except:
		with tf.device(get_available_device()):
			model = load(attr)
			data = get_data(attr=attr)
			#print(f'locals: {locals().keys()}')
			print(f'names in data: {data.keys()}')
			#print(f'y_test has shape {y_test.shape}')
		
			x_test = data['x_test']
			x_train = data['x_train']
			x_valid = data['x_valid']
			# Kludge to keep Keras (or Tensorflow?) from resetting the model weights.
			# model.fit(x_train[0:64,:,:,:], y_train[0:64,:],
			#		   batch_size=batch_size,
			#		   epochs=epochs,
			#		   validation_data=(x_test, y_test),
			#		   shuffle=True)
			#print('x_test in locals():', ('x_test' in locals()))
			"""scores = model.evaluate(x_test, y_test, verbose=1)
			print('Test loss:', scores[0])
			print('Test accuracy:', scores[1])"""
			
			# compute the class probabilities.
			print('predicting class probabilities')
			y_pred = model.predict(x_test, verbose=1)
		
		print('predicting classes')
		if MULTI or ('+' in attr):
			if MULTI_AS_ONEHOT_DISTINCT_CLASSES:
				y_pred = np.argmax(y_pred, axis=1)
				#y_test = np.argmax(y_test, axis=1)
			else:
				bin = pickle_load(binarizer_path.format(attr+'__multi'))
				print(y_pred)
				print(y_test)
				y_pred = np.array(bin.inverse_transform(y_pred)).astype(object).sum(1)
				#y_test = np.array(bin.inverse_transform(y_test)).astype(object).sum(1)
				classes = list(iter_unique(y_pred, y_test))
				for i,c in enumerate(classes):
					y_pred[y_pred==c] = i
					y_test[y_test==c] = i
				#y_pred_class_inds = np.argmax(y_pred, axis=1)
				#y_test_class_inds = np.argmax(y_test, axis=1)
				y_pred = y_pred.astype(int)
				#y_test = y_test.astype(int)
		else:
			# convert the probabilities into class labels.  in this model, the label 0 means female and 1 means male.	  
			y_pred = np.argmax(y_pred, axis=1)
			
			# convert the one-hot encoding of the test cases back into class labels.
		
		np.save(f'y_pred/y_pred_{attr}.npy', y_pred)
	
	print('reconverting labeled classes')
	y_test = np.argmax(y_test, axis=1)
	print('assessing misclassification')
	# determine where prediction and truth do not agree.
	I = np.where(np.not_equal(y_pred, y_test))[0]
	text = 'image {im} ({im2:06d}.jpg): predicted <{p}>, actual <{a}>'
	path = misclass_text_path.format(attr=attr)
	pathdir = output_img_dir.format(attr=attr)
	if not os.path.isdir(pathdir): os.mkdir(pathdir)
	values = iter_unique(y_pred,y_test)
	matrix = confusion_matrix(y_test,y_pred)
	matrix = (matrix*(100/np.sum(matrix))).round(2)
	with open(path,'w') as f:
		f.write(str(matrix))
		f.write('\n')
		for i,true in enumerate(values):
			for j,pred in enumerate(values):
				f.write(f'true:{values[i]} pred:{values[j]} {matrix[i][j]}\n')
		
		f.write('\n'.join(text.format(im=i,im2=i+num_train+num_valid+1,p=y_pred[i],a=y_test[i]) for i in I))

	# save the misclassified instances for further investigation. shift I
	# so that the numbers we restore correspond to the misclassified image.
	I_adj = I.astype(int) + (num_train + num_valid)
	
	print('saving misclassifications')
	np.save(misclass_path.format(attr=attr),I_adj)
	np.save(misclass_matrix_path.format(attr=attr),matrix)
	
	if MULTI:
		print(f'multi: y_pred (len={len(y_pred)}): {y_pred}')
		print(f'multi: y_test (len={len(y_test)}): {y_test}')
		i=0
		prediction_matrix = np.full(values.shape*2,-1,dtype=int)
		while i<len(y_pred):
			if not np.any(-1==prediction_matrix):
				print('the prediction matrix is full',flush=True)
				break
			if -1==prediction_matrix[y_test[i]][y_pred[i]]:
				prediction_matrix[y_test[i]][y_pred[i]] = i + (num_train + num_valid)
				print(
					f'set element at position {i}:({y_test[i]},{y_pred[i]}) in '
					'prediction_matrix:{prediction_matrix[y_test[i]][y_pred[i]]}',
					flush=True
				)
			i+=1
		np.save(
			misclass_prediction_matrix_path.format(attr = attr),
			prediction_matrix
		)

	# use indices in I to access the misclassified instances: y_test[I]
	return I_adj

if __name__=='__main__':
	from sys import argv
	get_misclassifications(argv[1])
	
