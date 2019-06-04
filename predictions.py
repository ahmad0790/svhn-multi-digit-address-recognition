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
from tensorflow.python.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import backend as K
from load_data import load_images
from cnn import build_cnn_model
from collections import defaultdict
#from keras.models import Model, load_model

def create_sliced_dataset(image, stride, window_size, x_scale, y_scale, counter):
	images_list = []
	slice_info = {}
	for x in range(0,image.shape[1],stride):
		for y in range(0,image.shape[0],stride):
			image_slice = image[y:y+window_size[1], x:x+window_size[0]]

			
			if image_slice.shape[0] == window_size[1] and image_slice.shape[1] == window_size[0] and image_slice.shape[0]>2 and image_slice.shape[1]>2:
				image_slice =cv2.resize(image_slice, (54,54))
				
				images_list.append(image_slice)
				slice_info[str(counter)] = {}
				slice_info[str(counter)]['x_scale'] = x_scale
				slice_info[str(counter)]['y_scale'] = y_scale
				slice_info[str(counter)]['x_coord'] = x
				slice_info[str(counter)]['y_coord'] = y
				counter +=1

	return np.array(images_list), slice_info, counter

def create_image_pyramids(image, stride, new_size):
	iteration = 0
	counter = 0
	##range used for video predictions. Bigger scale range because theree are more images
	scale_range = [1.5,1.2,1.0,0.7,0.5,0.3,0.2,0.1]

	##range used for image predictions
	#scale_range = [1.5,1.0,0.5,0.3]
	#x_scale_range = [0.2, 0.5]
	#y_scale_range = [0.5, 1.5]

	for x_scales in scale_range:
		for y_scales in scale_range:

			iteration += 1
			resized =cv2.resize(image, None, fx=x_scales,fy=y_scales)
			resized_images_list, image_info, counter = create_sliced_dataset(resized, stride, (new_size,new_size), x_scales, y_scales, counter)
			if iteration == 1:
				images_list = np.array(resized_images_list)
				resized_images_info = image_info.copy()
			else:

				if resized_images_list.shape[0]>0:
					images_list = np.concatenate((images_list,resized_images_list), axis=0)
					resized_images_info.update(image_info)


	return images_list, resized_images_info

def infer_best_match(images_list, model_predictions):
	len_max_indices = np.argmax(model_predictions[5], axis = 1)
	len_max_probs = np.max(model_predictions[5], axis = 1)

	best_length_probs = np.argsort(len_max_probs*-1)

	predicted_prob_sequence= []
	all_sequences = []
	sequence_probs =[]

	for i in range(0,images_list.shape[0]):

		pred_length = len_max_indices[i]
		total_prob = 1.00
		sequence = []
		prob_sequence = []

		for j in range(0,pred_length):

			sequence.append(np.argmax(model_predictions[j][i,:]))
			total_prob = total_prob+np.max(model_predictions[j][i,:])
			prob_sequence.append(np.max(model_predictions[j][i,:]))


		if pred_length >0 and pred_length <= 4:
			total_prob = total_prob+model_predictions[5][i,pred_length]

		else:
			total_prob = 0.0
		
		prob_sequence.append(model_predictions[5][i,pred_length])
		predicted_prob_sequence.append(total_prob)
		sequence_probs.append(prob_sequence)
		all_sequences.append(sequence)

	predicted_prob_sequence = np.array(predicted_prob_sequence)
	sequence_probs = np.array(sequence_probs)
	all_sequences = np.array(all_sequences)

	best_predictions = np.argsort(predicted_prob_sequence*-1)
	best_predictions_prob = np.sort(predicted_prob_sequence*-1)
	best_predictions_length = len_max_indices[best_predictions]
	best_sequences = all_sequences[best_predictions]
	best_individual_probs = sequence_probs[best_predictions]

	return best_predictions, best_predictions_prob, best_predictions_length, best_sequences, best_individual_probs