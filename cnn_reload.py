"""Simple script to load a model and evaluate it on training/test/validation sets."""

import tensorflow as tf
from cnn_util import *

from sys import argv
if len(argv)>1: attr=argv[1]
else: attr=attribute



with tf.device(get_available_device()):
	globals().update(get_data(attr=attr))
	model = load(attr)
	print('test:',model.evaluate(x_test,y_test,verbose=1))
	print('valid:',model.evaluate(x_valid,y_valid,verbose=1))
	print('train:',model.evaluate(x_train,y_train,verbose=1))