from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from skimage import transform, io
from scipy import misc
from skimage.color import rgb2gray

import cv2

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
		'model_dir', '/tmp/imagenet',
		"""Path to classify_image_graph_def.pb, """
		"""imagenet_synset_to_human_label_map.txt, and """
		"""imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '/tmp/imagenet/traffic.jpeg',
													 """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 3,
														"""Display this many predictions.""")


# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 64*64*3 # MNIST data input (img shape: 64*64)
n_classes = 2 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 64, 64, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 16*16*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([16*16*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, "modelCNN.ckpt")


def run_inference_on_image(image_data):
	yval = sess.run([pred], feed_dict={x: [image_data.flatten()] ,keep_prob: 1.})
	# print(yval)
	return yval

def is_car(image):
	return run_inference_on_image(image)

##########################

def equate(arr,n,m,startx,starty):
	tmp1 = np.zeros((n,m,3))
	for i in range(n):
		for j in range(m):
			tmp1[i][j][:]=arr[startx+i][starty+j][:]
	return tmp1

img=None
def image_parser(image):
	global img
	traffic_image = io.imread(image)
	img=traffic_image.copy()
	img = cv2.medianBlur(img,5) 
	img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
	ret,thresh = cv2.threshold(img,127,255,0)
	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	count=0
	print("total candidates:",len(contours))
	car_present = np.zeros((len(contours)))
	count=-1
	for cnt in contours:
		count+=1
		x,y,w,h = cv2.boundingRect(cnt)
		if (w<=20) or (h<=20):
			continue;
		patch=equate(traffic_image, h, w, y, x)
		patch=transform.resize(patch, (64,64))
		car_prob =is_car(patch)
		if(car_prob):
			car_present[count]=1

	to_take = np.zeros((len(contours)))
	count=-1
	for cnt in contours:
		count+=1
		if(car_present[count]==1):
			to_take[count]=1
			x,y,w,h = cv2.boundingRect(cnt)
			count2=-1
			for cnt2 in contours:
				count2+=1
				x2,y2,w2,h2 = cv2.boundingRect(cnt2)
				if(count==count2):
					continue
				if(x<=x2 and y<=y2 and x+w>=x2+w2 and y+h>=y2+h2 and car_present[count2]==1):
					to_take[count]=0

	count=-1
	for cnt in contours:
		count+=1
		if(to_take[count]==0):
			continue
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(traffic_image,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.imshow('boxes',traffic_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	n = traffic_image.shape[0]
	m = traffic_image.shape[1]
	count=0
	print(count)

def main(_):
    # vidcap = cv2.VideoCapture('.mp4')
	image_parser(sys.argv[1])

if __name__ == '__main__':
	tf.app.run()