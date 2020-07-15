import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import math
import random
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from render import Render
import pathlib
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def load_and_preprocess_image(path):
	image = tf.io.read_file(path)
	image = tf.image.decode_jpeg(image, channels=3)
	image = tf.cast(image, tf.float32) / 255.0
	return image

def rotate(image, angle):
	image = tfa.image.rotate(image, angle, interpolation='BILINEAR')
	return image

def crop(image, offset):
	image = tf.image.crop_to_bounding_box(image, int(offset[0]), int(offset[1]), 512, 512)
	return image

def normalize(n):
	n2 = n * 2 - 1 # (n - 0.5) * 2
	lengths = np.expand_dims(np.linalg.norm(n2, axis=-1), -1)
	n_norm = n2/lengths
	return n_norm

def gen_ds(root, img_name):
	data_root = root #'D:/Bin/images'
	data_root = pathlib.Path(data_root)
	paths = list(data_root.glob('*/{}.png'.format(img_name)))
	paths = [str(path) for path in paths]
	path_ds = tf.data.Dataset.from_tensor_slices(paths)
	img_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	return img_ds

def rectify_order(src_imgs, k):
	order = [19,18,17,16,23,22,21,20]
	if k == 0.00: order =  [19,18,17,16,23,22,21,20]
	if k == 0.125: order =  [20,19,18,17,16,23,22,21]
	if k == 0.25: order =  [21,20,19,18,17,16,23,22]
	if k == 0.375: order =  [22,21,20,19,18,17,16,23]
	if k == 0.5: order = [23,22,21,20,19,18,17,16]
	if k == 0.625: order = [16,23,22,21,20,19,18,17]
	if k == 0.75: order = [17,16,23,22,21,20,19,18]
	if k == 0.875: order = [18,17,16,23,22,21,20,19]
	seq = order + [x-8 for x in order] + [x-16 for x in order]

	temp = tf.identity(src_imgs)
	new_src = []
	for i in range(24):
		new_src.append(temp[seq[i],...])
	new_src = tf.stack(new_src, axis=0)
	return new_src


class Dataset2:
	def __init__(self, root, dstype=0, pad=256):
		t = [gen_ds(root, 'diffuse'), gen_ds(root, 'specular'), gen_ds(root, 'normal'), gen_ds(root, 'glossiness')]
		if dstype == 1:
			for i in range(24):
				t.append(gen_ds(root, 'P_L{}E'.format(i)))

		image_ds = tf.data.Dataset.zip(tuple(t))
		# image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		image_ds = image_ds.shuffle(buffer_size=20)
		image_ds = image_ds.repeat()
		# image_ds = image_ds.batch(1)
		image_ds = image_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		self.image_ds = iter(image_ds)
		self.root = root
		self.dstype = dstype
		self.pad = pad
		self.render = Render()

	# augment_type = 0: rotate at 4 fixed angles
	def nxt(self):
		maps = next(self.image_ds)
		# print(maps.shape)
		d = maps[0]
		s = maps[1]
		n = maps[2]
		g = maps[3]
		src = None
		if self.dstype == 1:
			src = tf.stack(maps[4:], axis=0)

		k = random.uniform(0,1)
		rand_rotate = lambda i: rotate(i, np.pi * 2 * k)

		offset = (random.uniform(self.pad, self.pad + 511), random.uniform(self.pad, self.pad + 511))
		rand_crop = lambda i: crop(i, offset)

		k_discrete = random.choice([i * 0.25 for i in range(4)])
		fixed_rotate = lambda i: rotate(i, np.pi * 2 * k_discrete)

		img_pad_map = lambda i: tf.pad(i, ((self.pad, self.pad), (self.pad, self.pad), (0,0)), "SYMMETRIC")
		# img_pad_src = lambda i: tf.pad(i, ((0,0), (pad_width, pad_width), (pad_width, pad_width), (0,0)), "SYMMETRIC")
		# rand rotate rand cut
		if self.dstype == 0:
			d,s,n,g = list(map(img_pad_map, (d,s,n,g)))
			d,s,n,g = list(map(rand_rotate, (d,s,n,g)))
			d,s,n,g = list(map(rand_crop, (d,s,n,g)))

		# fix rotate (include 24 src)
		if self.dstype == 1:
			d,s,n,g, src = list(map(fixed_rotate, (d,s,n,g, src)))
			# d,s,n,g = list(map(img_pad_map, (d,s,n,g)))
			# src = img_pad_src(src)
			src = rectify_order(src,k_discrete)


		# d = pow(d, 2.2)
		# s = pow(s, 2.2)
		# d = pow(d, 1.2)
		# s = pow(s, 2.2)
		n = normalize(n)
		g = tf.image.rgb_to_grayscale(g)

		# n = n * 2 - 1 # (n - 0.5) * 2
		# lengths = np.expand_dims(np.linalg.norm(n, axis=-1), -1)
		# n = n/lengths

		if self.dstype == 1:
			src = tf.transpose(src, [1, 2, 0, 3])
		if self.dstype == 0:
			lint = random.uniform(0.16,0.17)
			print(lint)
			lightInt = [[lint, lint, lint]] * 24
			src = self.render.render({'diffuse': d, 'specular': s, 'glossiness': g, 'normal': n}, lightInt=lightInt)

		return d, s, n, g, src
		# return d, s, n, g, src


if __name__ == '__main__':
	key = None
	ith = 0

	# ds1 = Dataset2('../pbr/pbr', dstype=0)
	ds = Dataset2('../____v2', dstype=0)
	# ds = [ds1, ds2]

	# ds = Dataset2('D:/Bin/images/', dstype=1)
	d,s,n,g, img24 = ds.nxt()
	# switch = 1

	while key != ord('z'):
		if key == ord(' '): 
			# if switch:
			# 	d,s,n,g, img24 = ds[0].nxt()
			# 	switch = 0
			# else:
			# 	d,s,n,g, img24 = ds[1].nxt()
			# 	switch = 1
			d,s,n,g, img24 = ds.nxt()

		if key == ord('n'):
			ith += 1
			ith = ith % 24

		# print(n)
		cv2.imshow('diffuse', cv2.cvtColor(d.numpy(), cv2.COLOR_RGB2BGR))
		cv2.imshow('specular', cv2.cvtColor(s.numpy(), cv2.COLOR_RGB2BGR))
		cv2.imshow('normal', cv2.cvtColor(n.numpy() * 0.5 + 0.5, cv2.COLOR_RGB2BGR))
		cv2.imshow('glossiness', g.numpy())
		# print(src.shape)
		# if ds.dstype:
		cv2.imshow('img24', cv2.cvtColor(img24[..., ith,:].numpy(), cv2.COLOR_RGB2BGR))

		key = cv2.waitKey(0)


