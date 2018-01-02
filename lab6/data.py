

def get_data(n=1342,m=175):
	from skimage.io import imread
	import numpy as np
	from skimage import transform
	from tqdm import trange, tqdm
	from random import shuffle
	import os

	input_dir = './cancer_data/inputs/'
	output_dir = './cancer_data/outputs/'

	files = os.listdir(input_dir)
	shuffle(files)
	train_images, train_label, test_images, test_label = [], [], [], []
	train_images_fn = [x for x in files if 'train' in x][:n]
	test_images_fn = [x for x in files if 'test' in x][:m]

	for f in tqdm(train_images_fn):
		train_images.append(transform.resize(imread(input_dir + f), (512,512,3), mode='constant'))
		train_label.append(transform.resize(imread(output_dir + f), (512,512), mode='constant'))

	for f in tqdm(test_images_fn):
		test_images.append(transform.resize(imread(input_dir + f), (512,512,3), mode='constant'))
		test_label.append(transform.resize(imread(output_dir + f), (512,512), mode='constant'))

	#whiten the data
	train_images = train_images - np.mean( train_images )
	train_images = train_images / np.sqrt( np.var( train_images ) )

	test_images = test_images - np.mean( test_images )
	test_images = test_images / np.sqrt( np.var( test_images ) )

	return train_images, np.array(train_label), test_images, np.array(test_label)