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
from load_data import load_images
from load_data_negatives import load_images as load_images_negative


#MODEL ARCHITECTURE BASED ON GOODFELLOW PAPER
def build_cnn_model():
	inputShape = (54, 54, 3)
	inputs = Input(shape=inputShape)

	x = Conv2D(48, (5, 5), padding="same")(inputs)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding="same", strides =(2,2))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(64, (5, 5), padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding="same", strides=(1,1))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(128, (5, 5), padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding="same", strides=(2,2))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(160, (5, 5), padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding="same", strides=(1,1))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(192, (5, 5), padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding="same",strides=(2,2))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(192, (5, 5), padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding="same", strides=(1,1))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(192, (5, 5), padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding="same", strides=(2,2))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(192, (5, 5), padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding="same",strides=(1,1))(x)
	#x = Dropout(0.25)(x)

	x = Flatten()(x)

	x = Dense(3072)(x)
	x = Activation("relu")(x)
	x = Dropout(0.25)(x)

	x = Dense(3072)(x)
	x = Activation("relu")(x)
	x = Dropout(0.25)(x)

	digit1 = Dense(11)(x)
	digit1 = Activation('softmax', name="digit1")(digit1)

	digit2 = Dense(11)(x)
	digit2 = Activation('softmax', name="digit2")(digit2)

	digit3 = Dense(11)(x)
	digit3 = Activation('softmax', name="digit3")(digit3)

	digit4 = Dense(11)(x)
	digit4 = Activation('softmax', name="digit4")(digit4)

	digit5 = Dense(11)(x)
	digit5 = Activation('softmax', name="digit5")(digit5)

	length = Dense(6)(x)
	length = Activation('softmax', name="length")(length)

	model = Model(inputs=inputs, outputs=[digit1, digit2, digit3, digit4, digit5, length], name="multidigit_classifier")

	return model

def all_categorical_accuracy(y_true, y_pred):
	print(y_true)
	print(y_pred)
	print(y_true.shape)
	print(y_pred.shape)
	return K.minimum(K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred[0], axis=-1)),K.floatx()),K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred[1], axis=-1)),K.floatx()))



#train_images, train_labels, train_length = load_images_negative(folder = 'train', augmentation = False)
#extra_images, extra_labels, extra_length = load_images_negative(folder = 'extra', augmentation = False)

#train_images, train_labels, train_length = load_images_negative(folder = 'train', augmentation = False, quick_load = True)
#extra_images, extra_labels, extra_length = load_images_negative(folder = 'extra', augmentation = False, quick_load = True)

#train_images = np.concatenate((train_images, extra_images), axis=0)
#train_labels = np.concatenate((train_labels, extra_labels), axis=0)
#train_length = np.concatenate((train_length, extra_length), axis=0)

#test_images, test_labels, test_length = load_images(folder = 'test', augmentation = False)
#test_images, test_labels, test_length = load_images(folder = 'test', augmentation = False, quick_load = True)

#print(train_images.shape)
#print(train_labels.shape)
#print(train_length.shape)

#print(test_images.shape)
#print(test_labels.shape)
#print(test_length.shape)

#np.save('train_images', train_images) 
#np.save('train_labels', train_labels) 
#np.save('train_length', train_length) 

#np.save('test_images', test_images) 
#np.save('test_labels', test_labels) 
#np.save('test_length', test_length) 
