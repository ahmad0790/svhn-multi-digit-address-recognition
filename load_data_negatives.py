import numpy as np
import cv2
import os
import copy
import math
import warnings
import tensorflow as tf
import h5py
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,Activation,MaxPooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

def get_box_data(index, hdf5_data):
	"""
	get `left, top, width, height` of each picture
	:param index:
	:param hdf5_data:
	:return:
	"""
	meta_data = dict()
	meta_data['height'] = []
	meta_data['label'] = []
	meta_data['left'] = []
	meta_data['top'] = []
	meta_data['width'] = []

	def print_attrs(name, obj):
		vals = []
		if obj.shape[0] == 1:
			vals.append(obj[0][0])
		else:
			for k in range(obj.shape[0]):
				vals.append(int(hdf5_data[obj[k][0]][0][0]))
		meta_data[name] = vals

	box = hdf5_data['/digitStruct/bbox'][index]
	hdf5_data[box[0]].visititems(print_attrs)
	return meta_data

def get_name(index, hdf5_data):
	name = hdf5_data['/digitStruct/name']
	return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

def load_images(folder, img_shape=64, resize_shape = 54, crops =1, augmentation=False, quick_load = False):

	mat_data = h5py.File(os.path.join(folder, 'digitStruct.mat'))
	images_array = []
	names_array = []
	negative_images_array = []
	size = mat_data['/digitStruct/name'].size
	
	if quick_load == True:
		size = 100
	#crops = 5
	labels_array = np.zeros((size*crops,5,11),dtype=np.uint8)
	length_array = np.zeros((size*crops,1,6), dtype=np.uint8)
	augmented_labels_array = np.zeros((size*crops, 5,11))
	augmented_length_array = np.zeros((size*crops, 1,6))
	
	negative_label = np.zeros((1,5,11),dtype=np.uint8)
	negative_length = np.zeros((1,1,6),dtype=np.uint8)
	negative_labels_array = np.zeros((size*crops,5,11),dtype=np.uint8)
	negative_length_array = np.zeros((size*crops,1,6), dtype=np.uint8)

	for i in range(0,size):
		#print(i)
		
		#print('Read Image Name')
		if i%1000==0:
			print("Images Read: " + str(i))
		
		image_file = get_name(i, mat_data)
		#print('Image File: ' + str(image_file))
		image = cv2.imread(str(folder) + '/' + str(image_file))
		image = image.astype(np.float32)

		name_string = image_file.split(".")[0]
		name = [str(s) for s in name_string if s.isdigit()]
		name = ''.join(name)
		#print('Image Name: ' + str(name))

		#print('Get Box Data')
		box = get_box_data(i, mat_data)
		label_string = box['label']
		#print(label_string)

		l = []
		for j in range(0, 5):

			if j < len(label_string):

				x = label_string[j]

				if x != 10:
					labels_array[i,j,round(int(x))] = 1
					#print(labels_array[i,j,round(int(x))])
					l.append(x)
				elif x == 10:
					labels_array[i,j,0] = 1
					#print(labels_array[i,j,0])
					l.append(0)

			elif j >= len(label_string):
				labels_array[i,j,10] = 1

		#print(len(l))
		length_array[i,0,len(l)] = 1

		height = int(max(box['height']))
		width = int(max(box['width']))*len(l)
		top = min(box['top'])
		top = int(max(top,0))
		left = min(box['left'])
		left = int(max(left,0))

		#cv2.rectangle(image, (left, top), (left+width*len(l), top+height), (0, 0, 255), 2)
		scale = 0.15
		height_extra, width_extra = int(height*scale),int(width*scale)
		
		image = image [max(0,int(top-height_extra)):int(top + height + height_extra), max(0,int(left-width_extra)):int(left+width+width_extra),:]		
		
		#CREATE NEGATIVE IMAGE
		negative_image = image[max(0,int(top-img_shape)):max(2, int(top-1)), max(0,int(left-img_shape)):max(2, int(left - 1)),:]

		if negative_image.shape[0] > 0 and negative_image.shape[1] > 0:
			negative_image = cv2.resize(negative_image, (resize_shape,resize_shape))

		else:
			negative_image = np.full((resize_shape,resize_shape,3),255)

		#cv2.imshow('image', negative_image.astype(np.uint8))
		#cv2.waitKey(0)

		negative_images_array.append(negative_image)
		negative_labels_array[i,:,10] = 1
		negative_length_array[i,0,0] = 1

		##AUGMENTATION
		if augmentation == True:

			image = cv2.resize(image, (img_shape,img_shape))

			for k in range(0,crops):
				x = np.random.randint(0, 10)
				y = np.random.randint(0, 10)
				random_crop = image[y:y+resize_shape, x:x+resize_shape]
				images_array.append(random_crop)
				names_array.append(random_crop)
				
				augmented_labels_array[i*crops+k,:,:] = labels_array[i,:,:]
				augmented_length_array[i*crops+k,:,:] = length_array[i,:,:]

				#print(l)
				#print(augmented_labels_array[i+k,:,:])
				#print(augmented_length_array[i+k,:,:])
				#cv2.imshow('image', random_crop.astype(np.uint8))
				#cv2.waitKey(0)

		else:
			image = cv2.resize(image, (resize_shape,resize_shape))
			images_array.append(image)

	#END OF LOOP TO ITERATE OVER ALL IMAGES
	images_array = np.array(images_array + negative_images_array)

	if augmentation == False:	
		#negative_labels_array = negative_labels_array.reshape(len(negative_images_array),5,11)
		#negative_length_array = negative_length_array.reshape(len(negative_length_array),1,6)
		
		#print(negative_length_array.shape)
		#print(negative_labels_array.shape)
		
		augmented_labels_array = np.concatenate((labels_array, np.array(negative_labels_array)), axis=0)
		augmented_length_array = np.concatenate((length_array, np.array(negative_length_array)), axis=0)

	else:
		augmented_labels_array = np.concatenate((augmented_labels_array, np.array(negative_labels_array)),axis=0)
		augmented_length_array = np.concatenate((augmented_length_array, np.array(negative_length_array)),axis=0)


	#print(augmented_length_array[995:1005,:,:])
	#print(augmented_labels_array[995:1005,:,:])

	return (images_array, augmented_labels_array, augmented_length_array)
