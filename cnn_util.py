import subprocess, sys

REMOTE_DIR = ... # remote location of python scripts

# ssh should already be configured the remote machine for this to work

if __name__=='__main__':
	subprocess.run(f'scp *.py {REMOTE_DIR}', shell=True)
	sys.exit()

import os, time, re, pickle
from functools import reduce
import numpy as np
import pandas as pd
import numexpr as ne
import pynvml as nv
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

INITIAL = set(locals().keys())

from cnn_params import *
from p2_entries import *

# email params

def mail(*, subject=MAIL_SUBJECT, message=DEFAULT_MESSAGE):
	subprocess.run(
		'echo "{m}" | mail -s "{s}" {a}'.format(
			m=message,s=subject,a=EMAIL
		),
		shell=True
	)

"""def get_run_attribute(attr=None):
	if attr is None:
		if MULTI: name = '+'.join(attributes)
		else: name = attribute
	else:
		if isinstance(attr,str): name = attr
		else: name = '+'.join(attr)
	return name"""

def get_tf_context():
	return tf.device(get_available_device())

def __load(attr):
	path = os.path.join(save_dir,attr+'.h5')
	print('loading the model... ', end='', flush=True)
	model = load_model(path)
	print('loaded from {}'.format(path), flush=True)
	return model

def load(attr, gpu = False):
	#name = get_run_attribute(attr)
	assert (attr is not None)
	
	if gpu: # run on the gpu
		with get_tf_context(): return __load(attr)
	
	return __load(attr)

def get_available_device(args=[], init=True):
	"""Convenience function that gets available GPU units and returns a string
	on the pattern f"/GPU:{i}" telling the index of the one currently using
	the lowest memory. Also sets the environment variable 'CUDA_VISIBLE_DEVICES'
	to f'{i}'.
	
	If there is an NVMLError in the attempt to get this information,
	`i` defaults to a pre-selected unit.
	
	If `args` has len > 1, the second argument is the integer index of the GPU.
	"""
	#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
	if init: nv.nvmlInit()
	if len(args)<=1:
		try:
			devices = map(nv.nvmlDeviceGetHandleByIndex, range(nv.nvmlDeviceGetCount()))
			#ind,device
			devices_enum = sorted(
				enumerate(devices),
				key = lambda d: nv.nvmlDeviceGetMemoryInfo(d[1]).free,
				reverse=True
			)
			ind = devices_enum[0][0]
		except nv.NVMLError as e:
			print(e)
			print(">  >  > error occurred: defaulting to gpu3")
			ind = 3
	else:
		ind = args[1]
	
	print('*\t*\t*\t*\t*\t*\t*\tget_available_device(): using device',ind)
	
	os.environ['CUDA_VISIBLE_DEVICES']=f'{ind}'
	
	return '/GPU:%i'%ind

def pickle_save(obj, f):
	with open(f,'wb') as w:
		pickle.dump(obj,w)

def pickle_load(f):
	with open(f,'rb') as w:
		result = pickle.load(w)
	return result

def __get_data(attr=None, short=False, raw=False, mem=True, source=False):
	"""Get training, test, and validation data for X and y, handling
	the logic of retrieving single-label vs. multi-label data."""
	if not source:
		# Read the labels, and split them between training, validation, and test sets.
		
		print('reading the labels... ', end='', flush=True)
		df = pd.read_csv(labels_path)
		print('done!', flush=True)
		if raw: return df
		if attr:
			if 'from' in attr: y = df[re.search('(.+)_from_(.+)',attr).groups()[0]]
			elif '+' in attr: y = df[list(re.search('(.+)\+(.+)',attr).groups())]
			else: y = df[attr]
		#elif MULTI: y = df[list(attributes)]
		#else: y = df[attribute]
		
		assert (attr is not None)
		
		# Convert the class vectors to binary class matrices.
		if '+' in attr:
			"""
			two options:
				I. multi-labeled matrix (each observation has multiple columns set to 1)
				II. expanded categories (each observation has one column set to 1)
			"""
			print('extracting multi-label data via one-hot-encoding... ', end='', flush=True)
			
			if MULTI_AS_ONEHOT_DISTINCT_CLASSES:
				
				#this is option II
				multilabels = np.tile(y.columns,(y.shape[0],1)).astype(object)
				multilabels += df[y.columns].values.astype(str)
				multilabels = multilabels.sum(axis=1)
				
				lb = LabelBinarizer()
				lb.fit(np.unique(multilabels))
				
				y = lb.transform(multilabels)
				print()
				print(multilabels)
				print(y)
				
				pickle_save(lb, binarizer_path.format(attr))
			
			else:
				#this is option I
				labels=[]
				for a in y.columns:
					l = a+'{}'
					for v in df[a].unique():
						labels.append(l.format(v))
				
				bin = MultiLabelBinarizer()
				bin.fit([labels])
				
				multilabels = np.tile(y.columns,(y.shape[0],1)).astype(object)
				multilabels += df[y.columns].values.astype(str)
				
				y = bin.transform(multilabels)
				pickle_save(bin, binarizer_path.format(attr+'__multi'))
			
			
		else:
			print('extracting label data via one-hot-encoding... ', end='', flush=True)
			y = to_categorical(y)
		
		y_train = y[:num_train]
		y_valid = y[num_train:num_train+num_valid]
		y_test  = y[num_train+num_valid:]
		
		print('done!', flush=True)
		if short: return sublocals(locals(),'y_train','y_test','y_valid')
	
	# Read the images, and split them between training, validation, and test sets.
	print('reading the scaled features... ', end='', flush=True)
	X = np.load(scaled_data_path)
	
	x_train = X[:num_train,:]
	x_valid = X[num_train:num_train+num_valid,:]
	x_test  = X[num_train+num_valid:,:]
	
	
	print('--------------------------------------------')
	print(x_train.shape[0], 'training samples')
	print('x_train shape:', x_train.shape)
	print(x_valid.shape[0], 'validation samples')
	print('x_valid shape:', x_valid.shape)
	print(x_test.shape[0],  'test samples')
	print('x_test shape:', x_test.shape)
	if source: return sublocals(locals(),'x_train','x_test','x_valid')
	print('y_train.shape:', y_train.shape)
	print('y_valid.shape:', y_valid.shape)
	print('y_test.shape:', y_test.shape)
	print('--------------------------------------------')
	return sublocals(locals(),'x_train','x_test','x_valid','y_train','y_test','y_valid')

def get_data(*, gpu=False, **kw):
	"""Get training, test, and validation data. Wrapper function
	that allows for setting the computational target."""
	if gpu:
		with tf.device(get_available_device()):
			return __get_data(**kw)
	else:
		return __get_data(**kw)

def get_all_attributes():
	return sorted(get_data(raw=True).columns)

def sublocals(locals,*attrs):
	print('sublocals: returning names',attrs)
	return {prop:locals[prop] for prop in attrs}

def class_matrix(*attrs):
	df = get_data(raw=True)[list(attrs)]
	for cl,count in zip(*np.unique(df.values,axis=0,return_counts=True)):
		print(cl,count)

def summarize(model):
	s = []
	model.summary(print_fn = lambda l: s.append(l))
	return '\n'.join(s)

def iter_unique(*s):
	return reduce(lambda a,b: np.unique(np.concatenate([np.unique(a),np.unique(b)])),s)

def write_model_run(hist, model, name, callback, subject):
	runfile = runpath.format(name)
	df = pd.DataFrame(index=['train','test','valid'], columns=['loss','accuracy'])
	
	with open(runfile, 'a') as f:
		pos = f.tell()
		#f.write(df.to_string())
		f.write(f'{subject}\n')
		f.write(f'times: {callback.get_times().round(1)}\n')
		f.write(pd.DataFrame(hist.history).to_string())
		f.write('\n')
		f.write(str(hist.params))
		f.write('\n')
		f.write(summarize(model))
		f.write('\n')
		#f.seek(pos)
		#text = str(f.read())
		f.write('\t'.join("*" for _ in range(8)))
		f.write('\n\n')

class CallbackMailer(keras.callbacks.Callback):
	"""Send an email whenever the model encounters certain events."""
	def __init__(self,subject):
		self.subject = subject
		self.message = '{a} epoch {e}'
		self.epoch_start_times = []
		self.epoch_end_times = []
	
	def mail(self, action, epoch):
		mail(subject = self.subject,
		     message = self.message.format(a=action, e=epoch+1))
	
	def on_epoch_begin(self, epoch, logs=None):
		self.mail('starting',epoch)
		self.epoch_start_times.append(time.time())
	
	def on_epoch_end(self, epoch, logs=None):
		self.mail('finishing',epoch)
		self.epoch_end_times.append(time.time())
	
	def get_times(self):
		return np.subtract(self.epoch_end_times , self.epoch_start_times)

__all__ = tuple(set(locals().keys()) - INITIAL)
