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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy') 
train_length = np.load('train_length.npy') 

test_images = np.load('test_images.npy') 
test_labels = np.load('test_labels.npy') 
test_length = np.load('test_length.npy') 

print(train_images.shape)
print(train_labels.shape)
print(train_length.shape)

print(test_images.shape)
print(test_labels.shape)
print(test_length.shape)

# initialize the optimizer and  the model
EPOCHS = 15
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

model = build_cnn_model()
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss=losses, metrics=["accuracy"])
print(model.summary())

callback = [EarlyStopping(monitor='val_loss', patience=3),
             ModelCheckpoint(filepath='best_cnn_model.h5', monitor='loss', save_best_only=True)]

history = model.fit(train_images,
	{"digit1": train_labels[:,0,:], "digit2": train_labels[:,1,:], "digit3": train_labels[:,2,:]
	,"digit4": train_labels[:,3,:],"digit5": train_labels[:,4,:],"length": train_length[:,0,:]},
	validation_data=(test_images[:,:,:,:], {"digit1": test_labels[:,0,:], "digit2": test_labels[:,1,:], "digit3": test_labels[:,2,:]
	,"digit4": test_labels[:,3,:],"digit5": test_labels[:,4,:],"length": test_length[:,0,:]}),
	epochs=EPOCHS,
	callbacks = callback,
	batch_size = BS,
	verbose=1,
	shuffle = True)

print(history.history.keys())

'''
# summarize history for accuracy
plt.plot(history.history['digit1_acc'])
plt.plot(history.history['val_digit1_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['digit1_loss'])
plt.plot(history.history['val_digit1_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''
