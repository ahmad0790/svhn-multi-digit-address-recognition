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
from predictions import create_image_pyramids, infer_best_match

OUT_DIR = "graded_images"
if not os.path.isdir(OUT_DIR):
	os.makedirs(OUT_DIR)

def save_image(filename, image):
	"""Convenient wrapper for writing images to the output directory."""
	cv2.imwrite(os.path.join(OUT_DIR, filename), image)

output_counter = 1

best_cnn_model = load_model('best_cnn_model.h5', custom_objects={'BatchNormalizationV1': BatchNormalization})
best_cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
img_list = ['cv-2464-161.png','cv-2464-717.png','cv-2464-981.png','cv-2464-1309.png','cv-2464-1448.png']

window= 40
stride = 10

for i in range(0, 5):

	print('Classification and Localization Results for Image: ' + str(i+1))

	image_file = img_list[i]
	image = cv2.imread(image_file).astype(np.float32)
	#image =cv2.resize(image, (240,120))

	images_list, resized_image_info = create_image_pyramids(image, stride = stride, new_size = window)
	print('Done Creating Reduced Scale Images')
	print('Number of Image Candidates: ' + str(images_list.shape[0]))

	model_predictions = best_cnn_model.predict(images_list)
	print("Model Predictions Complete")
	best_predictions, best_predictions_prob, best_predictions_length, best_sequences, best_individual_probs = infer_best_match(images_list, model_predictions)

	best_match = images_list[best_predictions[0]]
	#best_match_str = "best_match_" + "{}.png".format(output_counter)
	#cv2.imwrite(os.path.join(OUT_DIR, best_match_str), best_match)

	best_sequence = ''.join(map(str,best_sequences[0]))
	best_sequence_index = best_predictions[0]
	print('Best Digit Sequence: ' + str(best_sequence))
	#print(resized_image_info[str(best_predictions[0])])
	print('')

	actual_x = int(resized_image_info[str(best_sequence_index)]['x_coord']*1.00/resized_image_info[str(best_sequence_index)]['x_scale'])
	actual_y = int(resized_image_info[str(best_sequence_index)]['y_coord']*1.00/resized_image_info[str(best_sequence_index)]['y_scale'])

	box_x_length = int(window*1.00/resized_image_info[str(best_sequence_index)]['x_scale'])
	box_y_length = int(window*1.00/resized_image_info[str(best_sequence_index)]['y_scale'])

	cv2.rectangle(image, (actual_x, actual_y), (min(426, actual_x+box_x_length), min(239, actual_y+box_y_length)), (0,0,255), 2)
	cv2.putText(image, str(best_sequence), (actual_x, max(5, actual_y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)) 

	out_str = "{}.png".format(output_counter)
	save_image(out_str, image)
	output_counter += 1
