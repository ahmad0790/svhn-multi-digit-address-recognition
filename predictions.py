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
#from keras.models import Model, load_model

def infer_label(model_predictions):

	print('inferrirng')

	predicted_label = []
	predicted_length = []
	prediction_score = []
		
	length = model_predictions[5][0]
	predicted_length = np.argmax(length) + 1
	print(predicted_length)

	for j in range(0,predicted_length):
		preds = np.array(model_predictions[j][0])
		print(preds)
		predicted_label.append(np.argmax(preds))
		prediction_score.append(np.max(preds))

	return prediction_score, predicted_length, predicted_label

def create_sliced_dataset(image, stride, window_size):
	images_list = []
	for x in range(0,image.shape[1],stride):
		for y in range(0,image.shape[0],stride):
			#print(x,y)
			image_slice = image[y:y+window_size, x:x+window_size]

			
			if image_slice.shape[0] == window_size and image_slice.shape[1] == window_size:
				images_list.append(image_slice)

	return np.array(images_list)
			

print('Final Results')

image_file = 'test.png'
best_cnn_model = load_model('best_cnn_model.h5', custom_objects={'BatchNormalizationV1': BatchNormalization})
best_cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
image = cv2.imread(image_file).astype(np.float32)
#image = cv2.resize(image, (256,256))

images_list = create_sliced_dataset(image, 15, 54)
model_predictions = best_cnn_model.predict(images_list)

print(images_list.shape)
print(model_predictions[0].shape)
print(model_predictions[1].shape)
print(model_predictions[2].shape)
print(model_predictions[3].shape)
print(model_predictions[4].shape)
print(model_predictions[5].shape)

digit1_max_indices = np.argmax(model_predictions[0], axis = 1)
digit1_max_probs = np.max(model_predictions[0], axis = 1)

digit2_max_indices = np.argmax(model_predictions[1], axis = 1)
digit2_max_probs = np.max(model_predictions[1], axis = 1)

digit3_max_indices = np.argmax(model_predictions[2], axis = 1)
digit3_max_probs = np.max(model_predictions[2], axis = 1)

digit4_max_indices = np.argmax(model_predictions[3], axis = 1)
digit4_max_probs = np.max(model_predictions[3], axis = 1)

digit5_max_indices = np.argmax(model_predictions[4], axis = 1)
digit5_max_probs = np.max(model_predictions[4], axis = 1)

len_max_indices = np.argmax(model_predictions[5], axis = 1)
len_max_probs = np.max(model_predictions[5], axis = 1)

digit_indices = np.array([digit1_max_indices, digit2_max_indices, digit3_max_indices, digit4_max_indices, digit5_max_indices, len_max_indices]).reshape(6, images_list.shape[0])
digit_probs = np.array([digit1_max_probs, digit2_max_probs, digit3_max_probs, digit4_max_probs, digit5_max_probs, len_max_probs]).reshape(6, images_list.shape[0])

print(len_max_indices.shape)
print(len_max_indices)
print(len_max_probs)

print(digit_indices)
print(digit_probs)

predicted_prob_sequence = []

for i in range(0,images_list.shape[0]):

	total_prob = 0
	mean_prob = 0
	pred_length = digit_indices[5,i]
	
	if pred_length > 0:
		for j in range(0, pred_length):
			if digit_indices[j,i] != 10:
				total_prob += digit_probs[j,i]

		mean_prob = total_prob*1.00/pred_length

	else:
		total_prob = 0
		mean_prob = 0

	predicted_prob_sequence.append(total_prob)


predicted_prob_sequence = np.array(predicted_prob_sequence)

best_predictions = np.argsort(predicted_prob_sequence*-1)
best_predictions_prob = np.sort(predicted_prob_sequence*-1)
best_predictions_length = len_max_indices[best_predictions]

print(best_predictions[0:10])
print(best_predictions_length[0:10])

print(np.argmax(predicted_prob_sequence))
print(np.max(predicted_prob_sequence))

print(best_predictions[best_predictions_length==2][0:10])
print(best_predictions_length[best_predictions_length==2][0:10])

three_digit_preds = best_predictions[best_predictions_length==2][0:25]
print(three_digit_preds)

best_match = images_list[766]
print(best_match.shape)
cv2.imwrite("best_match.png", best_match)





#print(best_box)
#print(prediction_label)
#print(prediction_score)
#cv2.rectangle(image, (best_box[x], best_box[y]), (x+54, y+54), (0,0,255), 2)
cv2.imwrite("test_image.png", image)