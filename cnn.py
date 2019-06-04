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
from load_data_rotations import load_images as load_images_rot
from scipy.ndimage import rotate

#CNN MODEL ARCHITECTURE BASED ON GOODFELLOW PAPER
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
