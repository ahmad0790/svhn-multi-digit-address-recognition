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
from vgg_pretrained import build_vgg_custom, build_vgg_random, get_model_performance

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

train_images, train_labels, train_length = load_images_negative(folder = 'train', img_shape = 224, resize_shape =224,  augmentation = False, quick_load = True)
test_images, test_labels, test_length = load_images(folder = 'test', img_shape = 224, resize_shape = 224, augmentation = False, quick_load = True)

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
model = build_vgg_random()
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, metrics=["accuracy"])
print(model.summary())

callback = [EarlyStopping(monitor='val_loss', patience=EPOCHS),
             ModelCheckpoint(filepath='best_vgg_random_model.h5', monitor='val_loss', save_best_only=True)]

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

history_dict = history.history
model_performance = get_model_performance(history_dict, EPOCHS)
model_performance.to_csv('model_results_batch_best_vgg_random_32_lr_E4_D_025.csv', sep=',', header=True)
