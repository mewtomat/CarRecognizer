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

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

class NodeLookup(object):
	"""Converts integer node ID's to human readable labels."""
	uid_to_node_id={};
	node_id_to_uid={};
	uid_to_human={}
	def __init__(self,
							 label_lookup_path=None,
							 uid_lookup_path=None):
		if not label_lookup_path:
			label_lookup_path = os.path.join(
					FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
		if not uid_lookup_path:
			uid_lookup_path = os.path.join(
					FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
		self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

	def load(self, label_lookup_path, uid_lookup_path):
		"""Loads a human readable English name for each softmax node.

		Args:
			label_lookup_path: string UID to integer node ID.
			uid_lookup_path: string UID to human-readable string.

		Returns:
			dict from integer node ID to human-readable string.
		"""
		if not tf.gfile.Exists(uid_lookup_path):
			tf.logging.fatal('File does not exist %s', uid_lookup_path)
		if not tf.gfile.Exists(label_lookup_path):
			tf.logging.fatal('File does not exist %s', label_lookup_path)

		# Loads mapping from string UID to human-readable string
		proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
		# uid_to_human = {}
		p = re.compile(r'[n\d]*[ \S,]*')
		for line in proto_as_ascii_lines:
			parsed_items = p.findall(line)
			uid = parsed_items[0]
			human_string = parsed_items[2]
			self.uid_to_human[uid] = human_string

		# Loads mapping from string UID to integer node ID.
		# node_id_to_uid = {}
		proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
		for line in proto_as_ascii:
			if line.startswith('  target_class:'):
				target_class = int(line.split(': ')[1])
			if line.startswith('  target_class_string:'):
				target_class_string = line.split(': ')[1]
				self.node_id_to_uid[target_class] = target_class_string[1:-2]

		# uid_to_node_id={}
		for key,val in self.node_id_to_uid.items():
			self.uid_to_node_id[val] = key

		# Loads the final mapping of integer node ID to human-readable string
		node_id_to_name = {}
		for key, val in self.node_id_to_uid.items():
			if val not in self.uid_to_human:
				tf.logging.fatal('Failed to locate: %s', val)
			name = self.uid_to_human[val]
			node_id_to_name[key] = name

		return node_id_to_name

	def id_to_string(self, node_id):
		if node_id not in self.node_lookup:
			return ''
		return self.node_lookup[node_id]


def create_graph():
	"""Creates a graph from saved GraphDef file and returns a saver."""
	# Creates graph from saved graph_def.pb.
	with tf.gfile.FastGFile(os.path.join(
			FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')

node_lookup=None
req_uids=None

def initialise():
	global node_lookup
	global req_uids
	node_lookup=NodeLookup()
	file_required_uids = open("auto_wnids")
	req_uids=[]
	for line in file_required_uids:
		req_uids.append(line.strip())
	file_required_uids.close()
	create_graph()

def run_inference_on_image(image_data,req_uids):
	"""Runs inference on an image.

	Args:
		image: Image file name.

	Returns:
		Nothing
	"""
	with tf.Session() as sess:
		softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
		predictions = sess.run(softmax_tensor,{'DecodeJpeg:0': image_data})
		predictions = np.squeeze(predictions)
		top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
		for node in top_k:
			if NodeLookup.node_id_to_uid[node] in req_uids:
				return node_lookup.id_to_string(node), True;
		return node_lookup.id_to_string(top_k[0]),False;
		# car_prob=0
		# for uids in req_uids:
		#   if uids in NodeLookup.uid_to_node_id:
		#     car_prob = car_prob+predictions[NodeLookup.uid_to_node_id[uids]]
		# return car_prob


def maybe_download_and_extract():
	"""Download and extract model tar file."""
	dest_directory = FLAGS.model_dir
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>> Downloading %s %.1f%%' % (
					filename, float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
		print()
		statinfo = os.stat(filepath)
		print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
	tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def is_car(image):
	return run_inference_on_image(image, req_uids)
	# print(prob)
	# return (prob)

##########################

def equate(arr,n,m,startx,starty):
	# print(n,m)
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
	labels = ["" for x in range(len(contours))]
	count=-1
	for cnt in contours:
		count+=1
		x,y,w,h = cv2.boundingRect(cnt)
		if (w<=20) or (h<=20):
			continue;
		patch=equate(traffic_image, h, w, y, x)
		patch=transform.resize(patch, (100,100))
		label,car_prob =is_car(patch)
		if(car_prob):
			car_present[count]=1
			labels[count]=label

		# count+=1
	to_take = np.zeros((len(contours)))
	count=-1
	for cnt in contours:
		count+=1
		if(car_present[count]==1):
			print("hihi1 - ",count)
			to_take[count]=1
			x,y,w,h = cv2.boundingRect(cnt)
			count2=-1
			for cnt2 in contours:
				count2+=1
				x2,y2,w2,h2 = cv2.boundingRect(cnt2)
				if(count==count2):
					continue
				if(x<=x2 and y<=y2 and x+w>=x2+w2 and y+h>=y2+h2 and car_present[count2]==1):
					print("hihi2 - ",count)
					to_take[count]=0

	count=-1
	for cnt in contours:
		count+=1
		if(to_take[count]==0):
			continue
		print("hihi3 - ", count)
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.putText(traffic_image,labels[count],(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
		cv2.rectangle(traffic_image,(x,y),(x+w,y+h),(0,255,0),2)
	cv2.imshow('boxes',traffic_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	n = traffic_image.shape[0]
	m = traffic_image.shape[1]
	count=0

	print(count)


def main(_):
	maybe_download_and_extract()
	initialise()
	# is_car(FLAGS.image_file)
	image_parser(sys.argv[1])

if __name__ == '__main__':
	tf.app.run()