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

OUT_DIR = "frame_images"
if not os.path.isdir(OUT_DIR):
	os.makedirs(OUT_DIR)


###these are from our problem sets
def video_frame_generator(filename):
	"""A generator function that returns a frame on each 'next()' call.

	Will return 'None' when there are no frames left.

	Args:
		filename (string): Filename.

	Returns:
		None.
	"""
	video = cv2.VideoCapture(filename)

	while video.isOpened():
		ret, frame = video.read()

		if ret:
			yield frame
		else:
			break

	video.release()
	yield None

def only_read_video(video_name, input_video, fps, frame_ids, output_prefix,
							counter_init, predictions, window, stride):

	#video = os.path.join(VID_DIR, video_name)
	image_gen = video_frame_generator(video_name)

	image = image_gen.__next__()
	#print(image.shape)
	#image = cv2.resize(image, (427, 240))
	#print(image.shape)
	h, w, d = image.shape

	out_path = "output/cv_{}-{}".format(output_prefix[4:], video_name)
	#video_out = mp4_video_writer(out_path, (w, h), fps)

	output_counter = counter_init
	
	best_cnn_model = load_model('best_cnn_model.h5', custom_objects={'BatchNormalizationV1': BatchNormalization})
	best_cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			
	frame_num = 1

	while image is not None:
		#print(image.shape)
		#image = cv2.resize(image, (427, 240))
		#print(image.shape)

		##MODEL PREDICTIONS
		if predictions == True and frame_num >=60:
			print("Processing fame {}".format(frame_num))
			images_list, resized_image_info = preds.create_image_pyramids(image, stride, window)
			if output_counter == 60:
				print("Number of Images: " + str(images_list.shape))
			model_predictions = best_cnn_model.predict(images_list)
			best_predictions, best_predictions_prob, best_predictions_length, best_sequences, best_individual_probs = preds.infer_best_match(images_list, model_predictions)
			best_match = images_list[best_predictions[0]]
			best_match_str = "best_match_" + "{}.png".format(output_counter)
			cv2.imwrite(os.path.join(OUT_DIR, best_match_str), best_match)

			#print('The Best')
			#print(resized_image_info)
			#print(best_predictions[0])
			#print(best_predictions_prob[0])
			#print(best_predictions_length[0])
			#print(best_sequences[0])
			#print(best_individual_probs[0])

			best_sequence = ''.join(map(str,best_sequences[0]))
			best_sequence_index = best_predictions[0]
			print('Best Sequence: ' + str(best_sequence))
			print(resized_image_info[str(best_predictions[0])])
			print('')

			actual_x = int(resized_image_info[str(best_sequence_index)]['x_coord']*1.00/resized_image_info[str(best_sequence_index)]['x_scale'])
			actual_y = int(resized_image_info[str(best_sequence_index)]['y_coord']*1.00/resized_image_info[str(best_sequence_index)]['y_scale'])

			box_x_length = int(window*1.00/resized_image_info[str(best_sequence_index)]['x_scale'])
			box_y_length = int(window*1.00/resized_image_info[str(best_sequence_index)]['y_scale'])

			cv2.rectangle(image, (actual_x, actual_y), (min(426, actual_x+box_x_length), min(239, actual_y+box_y_length)), (0,0,255), 2)
			cv2.putText(image, str("Found Address: " + best_sequence), (actual_x, max(5, actual_y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)) 

			video_out.write(image)

		output_counter += 1

		out_str = output_prefix + "{}.png".format(output_counter)
		save_image(out_str, image)
		
		image = image_gen.__next__()
		frame_num += 1

	#video_out.release()


def process_video(video_name, input_video, fps, frame_ids, output_prefix,
							counter_init, predictions, window, stride):

	video = os.path.join(VID_DIR, video_name)
	image_gen = video_frame_generator(video_name)

	image = image_gen.__next__()
	#print(image.shape)
	#image = cv2.resize(image, (427, 240))
	#print(image.shape)
	h, w, d = image.shape

	out_path = "output/cv_{}-{}".format(output_prefix[4:], video_name)
	video_out = mp4_video_writer(out_path, (w, h), fps)

	output_counter = counter_init
	
	best_cnn_model = load_model('best_cnn_model.h5', custom_objects={'BatchNormalizationV1': BatchNormalization})
	best_cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
			
	frame_num = 1

	while image is not None:
		#print(image.shape)
		#image = cv2.resize(image, (427, 240))
		#print(image.shape)

		##MODEL PREDICTIONS
		if predictions == True and frame_num >=60:
			print("Processing fame {}".format(frame_num))
			images_list, resized_image_info = preds.create_image_pyramids(image, stride, window)
			if output_counter == 60:
				print("Number of Images: " + str(images_list.shape))
			model_predictions = best_cnn_model.predict(images_list)
			best_predictions, best_predictions_prob, best_predictions_length, best_sequences, best_individual_probs = preds.infer_best_match(images_list, model_predictions)
			best_match = images_list[best_predictions[0]]
			best_match_str = "best_match_" + "{}.png".format(output_counter)
			cv2.imwrite(os.path.join(OUT_DIR, best_match_str), best_match)

			#print('The Best')
			#print(resized_image_info)
			#print(best_predictions[0])
			#print(best_predictions_prob[0])
			#print(best_predictions_length[0])
			#print(best_sequences[0])
			#print(best_individual_probs[0])

			best_sequence = ''.join(map(str,best_sequences[0]))
			best_sequence_index = best_predictions[0]
			print('Best Sequence: ' + str(best_sequence))
			print(resized_image_info[str(best_predictions[0])])
			print('')

			actual_x = int(resized_image_info[str(best_sequence_index)]['x_coord']*1.00/resized_image_info[str(best_sequence_index)]['x_scale'])
			actual_y = int(resized_image_info[str(best_sequence_index)]['y_coord']*1.00/resized_image_info[str(best_sequence_index)]['y_scale'])

			box_x_length = int(window*1.00/resized_image_info[str(best_sequence_index)]['x_scale'])
			box_y_length = int(window*1.00/resized_image_info[str(best_sequence_index)]['y_scale'])

			cv2.rectangle(image, (actual_x, actual_y), (min(426, actual_x+box_x_length), min(239, actual_y+box_y_length)), (0,0,255), 2)
			cv2.putText(image, str("Found Address: " + best_sequence), (actual_x, max(5, actual_y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)) 


			out_str = output_prefix + "{}.png".format(output_counter)
			save_image(out_str, image)
			video_out.write(image)

		output_counter += 1
		
		image = image_gen.__next__()
		frame_num += 1

	video_out.release()


def mp4_video_writer(filename, frame_size, fps=30):
	"""Opens and returns a video for writing.

	Use the VideoWriter's `write` method to save images.
	Remember to 'release' when finished.

	Args:
		filename (string): Filename for saved video
		frame_size (tuple): Width, height tuple of output video
		fps (int): Frames per second
	Returns:
		VideoWriter: Instance of VideoWriter ready for writing
	"""
	fourcc = cv2.VideoWriter_fourcc(*'MP4V')
	return cv2.VideoWriter(filename, int(fourcc), 30, frame_size)


def save_image(filename, image):
	"""Convenient wrapper for writing images to the output directory."""
	cv2.imwrite(os.path.join(OUT_DIR, filename), image)

##note typo: actual image is 2464 address not 2468
frame_ids = [10,100,200]
fps = 30
video_file = "movie_2468.mp4"
my_video = "movie_2468.mp4"  
process_video(video_file, my_video, fps, frame_ids, "cv-2464-", 1, True, 40,10)

'''
video_file = "movie_2348.mp4"
my_video = "movie_2348.mp4"  
process_video(video_file, my_video, fps, frame_ids, "cv-2348-", 1, True, 40,5)

video_file = "movie_525.mp4"
my_video = "movie_525.mp4"  
process_video(video_file, my_video, fps, frame_ids, "cv-525-", 1, True, 40,10)

video_file = "movie_512.mp4"
my_video = "movie_512.mp4"  
process_video(video_file, my_video, fps, frame_ids, "cv-512-", 1, True, 40,10)
'''
