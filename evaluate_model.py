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
import predictions as preds

def predict_sequences(images_list, model_predictions):
	
	len_max_indices = np.argmax(model_predictions[5], axis = 1)
	all_sequences = []

	for i in range(0,images_list.shape[0]):
		pred_length = len_max_indices[i]
		sequence = []

		for j in range(0,5):
			if j < pred_length:
				sequence.append(np.argmax(model_predictions[j][i,:]))
			else:
				sequence.append(10)
		
		all_sequences.append(sequence)
	all_sequences = np.array(all_sequences)
	return all_sequences

def get_all_match_accuracy(true_sequences, predicted_sequences):
	true_sequences = np.argmax(true_sequences[:,:,:], axis=2)
	not_matched = np.sum(true_sequences - predicted_sequences, axis =1)

	all_predictions = np.zeros((not_matched.shape[0],))
	all_predictions[not_matched==0] = 1

	correctly_classified = np.sum(all_predictions, axis = 0)
	accuracy = np.float32(correctly_classified)*1.0000/np.float32(all_predictions.shape[0])*1.00

	return accuracy

def evaluate_model_accuracy(model_filename, test_images, test_labels):
	best_cnn_model = load_model(model_filename, custom_objects={'BatchNormalizationV1': BatchNormalization})
	best_cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model_predictions = best_cnn_model.predict(test_images)
	predicted_sequences = np.array(predict_sequences(test_images[:,:,:,:], model_predictions))
	acc = get_all_match_accuracy(test_labels[:,:,:], predicted_sequences)
	return acc


##these were trained on 600K data
test_images, test_labels, test_length = load_images(folder = 'test', augmentation = False, quick_load=False)
print('CNN Model Accuracy with Data Augmentation and Rotation')
cnn_rot_acc = evaluate_model_accuracy('best_cnn_model.h5', test_images, test_labels)
print(cnn_rot_acc)


test_images, test_labels, test_length = load_images(folder = 'test', augmentation = False, quick_load=False)
cnn_extra_acc = evaluate_model_accuracy('best_cnn_model_eextras.h5', test_images, test_labels)

print('CNN Model Accuracy with Extras Data Augmentation')
print(cnn_extra_acc)

##these weree trained on 66K data because of memory issues
test_images, test_labels, test_length = load_images(folder = 'test', img_shape = 224, resize_shape = 224, augmentation = False, quick_load = False)

print('VGG Model with Pretrained Weights Accuracy')
vgg_pre_acc = evaluate_model_accuracy('best_vgg_pre_model.h5',test_images, test_labels)
print(vgg_pre_acc)


print('VGG Model with Random Weights Accuracy')
vgg_random_acc = evaluate_model_accuracy('best_vgg_random_model.h5',test_images, test_labels)
print(vgg_random_acc)

#print("True Values")
#print(np.argmax(test_labels[0:10,:,:], axis=2))

#print("Predicted Values")
#print(predicted_sequences[0:10,:])
