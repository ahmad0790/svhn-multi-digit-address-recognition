import numpy as np
import cv2
import os
import copy
import math
import warnings
import tensorflow as tf
import h5py
import pandas as pd
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,Activation,MaxPooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint, History
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import backend as K
from load_data import load_images
from load_data_negatives import load_images as load_images_negative
from load_data_rotations import load_images as load_images_rot
from cnn import build_cnn_model
from tensorflow.python.keras.applications.vgg16 import VGG16

def get_model_performance(history_dict,epochs):

	model_performance_df = pd.DataFrame({'Epoch' : list(range(0, epochs))})

	model_performance_df['train_loss'] = history_dict["loss"]
	model_performance_df['digit1_loss'] = history_dict["digit1_loss"]
	model_performance_df['digit2_loss'] = history_dict["digit2_loss"]
	model_performance_df['digit3_loss'] = history_dict["digit3_loss"]
	model_performance_df['digit4_loss'] = history_dict["digit4_loss"]
	model_performance_df['digit5_loss'] = history_dict["digit5_loss"]
	model_performance_df['length_loss'] = history_dict["length_loss"]

	model_performance_df['val_loss'] = history_dict["val_loss"]
	model_performance_df['val_digit1_loss'] = history_dict["val_digit1_loss"]
	model_performance_df['val_digit2_loss'] = history_dict["val_digit2_loss"]
	model_performance_df['val_digit3_loss'] = history_dict["val_digit3_loss"]
	model_performance_df['val_digit4_loss'] = history_dict["val_digit4_loss"]
	model_performance_df['val_digit5_loss'] = history_dict["val_digit5_loss"]
	model_performance_df['val_length_loss'] = history_dict["val_length_loss"]

	model_performance_df['digit1_acc'] = history_dict["digit1_acc"]
	model_performance_df['digit2_acc'] = history_dict["digit2_acc"]
	model_performance_df['digit3_acc'] = history_dict["digit3_acc"]
	model_performance_df['digit4_acc'] = history_dict["digit4_acc"]
	model_performance_df['digit5_acc'] = history_dict["digit5_acc"]
	model_performance_df['length_acc'] = history_dict["length_acc"]

	model_performance_df['val_digit1_acc'] = history_dict["val_digit1_acc"]
	model_performance_df['val_digit2_acc'] = history_dict["val_digit2_acc"]
	model_performance_df['val_digit3_acc'] = history_dict["val_digit3_acc"]
	model_performance_df['val_digit4_acc'] = history_dict["val_digit4_acc"]
	model_performance_df['val_digit5_acc'] = history_dict["val_digit5_acc"]
	model_performance_df['val_length_acc'] = history_dict["val_length_acc"]

	return model_performance_df

#MODEL ARCHITECTURE
def build_vgg_custom():
	#MODEL ARCHITECTURE
	inputShape = (64, 64, 3)
	inputs = Input(shape=inputShape)

	x = Conv2D(64, (3, 3), padding="same", strides=(1,1))(inputs)
	x = Activation("relu")(x)
	x = Conv2D(64, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding ='same', strides = (2,2))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(128, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(128, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2),padding ='same', strides = (2,2))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(256, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(256, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(256, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding ='same', strides = (2,2))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(512, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(512, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(512, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding ='same', strides = (2,2))(x)
	#x = Dropout(0.25)(x)

	x = Flatten()(x)

	x = Dense(2048)(x)
	x = Activation("relu")(x)
	#x = Dropout(0.50)(x)

	x = Dense(2048)(x)
	x = Activation("relu")(x)
	#x = Dropout(0.50)(x)

	x = Dense(2048)(x)
	x = Activation("relu")(x)
	#x = Dropout(0.50)(x)

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

def build_vgg_random():
	#MODEL ARCHITECTURE
	inputShape = (224, 224, 3)
	inputs = Input(shape=inputShape)

	x = Conv2D(64, (3, 3), padding="same", strides=(1,1))(inputs)
	x = Activation("relu")(x)
	x = Conv2D(64, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding ='same', strides = (2,2))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(128, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(128, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2),padding ='same', strides = (2,2))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(256, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(256, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(256, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding ='same', strides = (2,2))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(512, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(512, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(512, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding ='same', strides = (2,2))(x)
	#x = Dropout(0.25)(x)

	x = Conv2D(512, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(512, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = Conv2D(512, (3, 3), padding="same", strides=(1,1))(x)
	x = Activation("relu")(x)
	x = MaxPooling2D(pool_size=(2, 2), padding ='same', strides = (2,2))(x)
	#x = Dropout(0.25)(x)

	x = Flatten()(x)

	x = Dense(4096)(x)
	x = Activation("relu")(x)
	#x = Dropout(0.50)(x)

	x = Dense(4096)(x)
	x = Activation("relu")(x)
	#x = Dropout(0.50)(x)

	x = Dense(4096)(x)
	x = Activation("relu")(x)
	#x = Dropout(0.50)(x)

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

def build_vgg_pretrained_model():
	inputShape = (224, 224, 3)
	inputs = Input(shape=inputShape)

	model_vgg16_pretrained = VGG16(weights='imagenet', include_top=False)
	#model_vgg16_pretrained.summary()

	conv_output = model_vgg16_pretrained(inputs)

	x = Flatten()(conv_output)

	x = Dense(4096)(x)
	x = Activation("relu")(x)
	x = Dropout(0.25)(x)

	x = Dense(4096)(x)
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

	print('New Model')
	print(model.summary())

	return model

if __name__ == "__main__":	

	sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

	#extra_images, extra_labels, extra_length = load_images(folder = 'extra', img_shape = 224, augmentation = False)
	#train_images = np.concatenate((train_images, extra_images), axis=0)
	#train_labels = np.concatenate((train_labels, extra_labels), axis=0)
	#train_length = np.concatenate((train_length, extra_length), axis=0)

	#rot_images, rot_labels, rot_length = load_images_rot(folder='train',img_shape = 224, resize_shape =224, crops =3, augmentation=True)
	train_images, train_labels, train_length = load_images_negative(folder = 'train', img_shape = 224, resize_shape =224,  augmentation = False, quick_load = False)
	#extra_images, extra_labels, extra_length = load_images_negative(folder = 'extra', img_shape = 224, resize_shape =224,  augmentation = False, quick_load = False)
	test_images, test_labels, test_length = load_images(folder = 'test', img_shape = 224, resize_shape = 224, augmentation = False, quick_load = False)

	#train_images = np.concatenate((train_images, extra_images), axis=0)
	#train_labels = np.concatenate((train_labels, extra_labels), axis=0)
	#train_length = np.concatenate((train_length, extra_length), axis=0)

	#train_images = np.concatenate((train_images, rot_images), axis=0)
	#train_labels = np.concatenate((train_labels, rot_labels), axis=0)
	#train_length = np.concatenate((train_length, rot_length), axis=0)

	'''
	np.save('train_images_vgg', train_images) 
	np.save('train_labels_vgg', train_labels) 
	np.save('train_length_vgg', train_length) 

	np.save('test_images_vgg', test_images) 
	np.save('test_labels_vgg', test_labels) 
	np.save('test_length_vgg', test_length) 

	train_images = np.load('train_images_vgg.npy')
	train_labels = np.load('train_labels_vgg.npy') 
	train_length = np.load('train_length_vgg.npy') 

	test_images = np.load('test_images_vgg.npy') 
	test_labels = np.load('test_labels_vgg.npy') 
	test_length = np.load('test_length_vgg.npy') 
	'''

	print(train_images.shape)
	print(train_labels.shape)
	print(train_length.shape)

	print(test_images.shape)
	print(test_labels.shape)
	print(test_length.shape)

	# initialize the optimizer and  the model
	EPOCHS = 20
	INIT_LR = 1e-4
	BS = 32


	losses = {
		"digit1": "categorical_crossentropy",
		"digit2": "categorical_crossentropy",
		"digit3": "categorical_crossentropy",
		"digit4": "categorical_crossentropy",
		"digit5": "categorical_crossentropy",
		"length": "categorical_crossentropy",
	}

	#model = build_vgg_custom()
	model = build_vgg_pretrained_model()
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(optimizer=opt, loss=losses, metrics=["accuracy"])


	callback = [EarlyStopping(monitor='val_loss', patience=EPOCHS),
	             ModelCheckpoint(filepath='best_vgg_pre_model.h5', monitor='val_loss', save_best_only=True)]

	#callback = [EarlyStopping(monitor='val_loss', patience=3)]

	history = model.fit(train_images,
		{"digit1": train_labels[:,0,:], "digit2": train_labels[:,1,:], "digit3": train_labels[:,2,:]
		,"digit4": train_labels[:,3,:],"digit5": train_labels[:,4,:],"length": train_length[:,0,:]},
		validation_data=(test_images[:,:,:,:], {"digit1": test_labels[:,0,:], "digit2": test_labels[:,1,:], "digit3": test_labels[:,2,:]
		,"digit4": test_labels[:,3,:],"digit5": test_labels[:,4,:],"length": test_length[:,0,:]}),
		epochs=EPOCHS,
		batch_size = BS,
		callbacks =callback,
		verbose=1,
		shuffle = True)

	print(history.history)
	history_dict = history.history

	model_performance = get_model_performance(history_dict, EPOCHS)
	model_performance.to_csv('model_results_batch_best_vgg_32_lr_E4_D_025.csv', sep=',', header=True)
