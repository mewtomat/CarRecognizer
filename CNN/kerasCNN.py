import os
import numpy as np
import pandas as pd
import keras
from skimage import data, io
from skimage import transform

from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D
from keras.layers import MaxPooling2D, Flatten, Dropout

BATCH_SIZE = 32
NB_EPOCH = 3

IMAGE_SIZE = 50

DATA_DIR = "resized_images"

def CNN():
	model = Sequential()
	# 50 x 50 x 3
	model.add(Convolution2D(8, 3, 3, border_mode='same',
		                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
	# 50 x 50 x 8
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
	# 25 x 25 x 8
	model.add(Convolution2D(16, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))
	model.add(Flatten())
	model.add(Dense(2))
	model.add(Activation('softmax'))
	model.compile(optimizer='adam', 
		          loss='binary_crossentropy',
		          metrics=['accuracy'])
	return model


# returns 50 x 50 x 3 iamge
def transform_img(image):
	return transform.resize(image, (IMAGE_SIZE,IMAGE_SIZE, image.shape[2]))


def loadData():
	images = os.listdir(DATA_DIR)
	train_data = []
	train_labels = []
	for image in images:
		if image[-4:] == 'jpeg':
			transformed_image = transform_img(io.imread(DATA_DIR + '/' + image))
			train_data.append(transformed_image)
			label_file = image[:-5] + '.txt'
			with open(DATA_DIR + '/' + label_file) as f:
				content = f.readlines()
				label = int(float(content[0]))
				l = [0, 0]
				l[label] = 1
				train_labels.append(l)
	return np.array(train_data), np.array(train_labels)


train_data, train_labels = loadData()
idx = np.random.permutation(train_data.shape[0])
model = CNN()
model.fit(train_data[idx], train_labels[idx], nb_epoch=NB_EPOCH)

preds = np.argmax(model.predict(train_data), axis=1)
train_labels = np.argmax(train_labels, axis=1)
print accuracy_score(train_labels, preds)