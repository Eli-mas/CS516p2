import os

num_train = 150000
num_valid =  25000
batch_size = 40
epochs = 4
data_augmentation = False

POOL=True
MULTI = True

MULTI_AS_ONEHOT_DISTINCT_CLASSES = True

save_dir = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
	print('made save directory:',save_dir)
#sys.exit()

rundir = 'runs/'
if not os.path.isdir(rundir):
	os.mkdir(rundir)
runpath = os.path.join(rundir,'runs_{}.txt')

combined_attribute_template = '{new}_from_{extant}'

if MULTI:
	attributes = sorted(['male','smiling'])
	final_activator = 'softmax' if MULTI_AS_ONEHOT_DISTINCT_CLASSES else 'sigmoid'
	loss = 'categorical_crossentropy' if MULTI_AS_ONEHOT_DISTINCT_CLASSES else 'binary_crossentropy'
	model_name = '+'.join(attributes) + '.h5'
else:
	# attribute = 'blond_hair'
	attribute = 'male'
	final_activator = 'softmax'
	loss = 'categorical_crossentropy'
	model_name = attribute + '.h5'
	
	# the attribute our classifier determines.
	new_attribute = 'blond_hair'
	new_combined_attribute = f'{new_attribute}_from_{attribute}'
	new_model_name = new_combined_attribute+'.h5'
	new_model_path = os.path.join(save_dir, new_model_name)

model_path = os.path.join(save_dir, model_name)

raw_model_name = '{attr}.h5'
raw_model_path = os.path.join(save_dir, raw_model_name)

#print('the model path is <{}>'.format(model_path))

binzarizer_dir = 'binarizers'
if not os.path.isdir(binzarizer_dir): os.mkdir(binzarizer_dir)
binarizer_path = os.path.join(binzarizer_dir,'{}.pickle')

master_dir = ... # source directory for image data
labels_path = f'{master_dir}/labels.csv'
data_path = f'{master_dir}/celeba.npy'
scaled_data_path='celebA_rescaled.npy'
img_dir = f'{master_dir}/images'

images_path = '{dir:s}/{index:06d}.jpg'
output_img_dir = 'gradcam_out/{attr}'
output_image_path = os.path.join(output_img_dir,'{type}_{index:06d}.jpg')
output_multi_image_path = os.path.join(output_img_dir,'{type}_matrix.pdf')
misclass_path = os.path.join(output_img_dir,'misclass_{attr}.npy')
misclass_matrix_path = os.path.join(output_img_dir,'misclass_matrix_{attr}.npy')
misclass_prediction_matrix_path = os.path.join(output_img_dir,'misclass_prediction_matrix_{attr}.npy')
misclass_text_path = os.path.join(output_img_dir,'misclass_{attr}.txt')

