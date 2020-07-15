import os, math, random, threading, queue, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import moderngl
import numpy as np
import tensorflow as tf

from svbrdf import *
from render import Render

from datetime import datetime
from dataset2 import Dataset2
# from render import *
# from loss import LossHelper
from network import SVBRDF_Net



if __name__ == '__main__':
	model = tf.keras.models.load_model('../model_0390.h5')
	render = Render()
	key, ith = None, 0

	gamma = 1.0

	lpos = [[61, 61, 207]]
	campos = [0, 0, 400]
	lint = 0.18



	# train_dataset = Dataset()
	# dataset_root = os.path.join(os.path.dirname(__file__), '..', 'pbr')
	# train_dataset.load(os.path.join(dataset_root, 'train.txt'))
	# train_dataset.start()
	# tf_train_dataset = train_dataset.get_tf_dataset().batch(1)
	# train_iter = iter(tf_train_dataset)
	# item = next(train_iter)
	# img24 = render.render(item, unflatten=False)
	# pred = model(tf.concat(img24, axis=-1), training=False)
	# print('d', item['diffuse'].shape)
	# print('s', item['specular'].shape)
	# print('g', item['glossiness'].shape)
	# print('n', item['normal'].shape)





	# ds = Dataset2('../pbr/test', dstype=0)
	# ds = Dataset2('D:/Bin/images/', dstype=1)

	ds = Dataset2('../____v2', dstype=0)
	# ds2 = Dataset2('../extra', dstype=1)
	# ds = [ds1, ds2]
	# switch = 1


	d,s,n,g, img24 = ds.nxt()
	# print('d', d.shape)
	# print('s', s.shape)
	# print('n', n.shape)
	# print('g', g.shape)
	svbrdf = {'diffuse': pow(d, gamma),'specular': s,'glossiness': g,'normal': n}
	img72 = tf.reshape(img24, list(img24.shape)[:-2] + [-1])
	pred = model(tf.expand_dims(img72, 0), training=False)




	pred_svbrdf = {
		'diffuse': pred[0],
		'specular': pred[1],
		'glossiness': pred[2],
		'normal': pred[3]
	}
	# print('pred g', pred[2].shape)
	while key != ord('z'):
		if key == ord(' '): 


			# item = next(train_iter)
			# img24 = render.render(item, unflatten=False)
			# pred = model(tf.concat(img24, axis=-1), training=False)
			# print('d', item['diffuse'].shape)
			# print('s', item['specular'].shape)
			# print('g', item['glossiness'].shape)
			# print('n', item['normal'].shape)




			# if switch:
			# 	d,s,n,g, img24 = ds[0].nxt()
			# 	switch = 0
			# else:
			# 	d,s,n,g, img24 = ds[1].nxt()
			# 	switch = 1
			d,s,n,g, img24 = ds.nxt()


			# print('d', d.shape)
			# print('s', s.shape)
			# print('n', n.shape)
			# print('g', g.shape)
			svbrdf = {'diffuse': pow(d, gamma),'specular': s,'glossiness': g,'normal': n}
			img72 = tf.reshape(img24, list(img24.shape)[:-2] + [-1])
			pred = model(tf.expand_dims(img72, 0), training=False)





			pred_svbrdf = {
				'diffuse': pred[0],
				'specular': pred[1],
				'glossiness': pred[2],
				'normal': pred[3]
			}

		if key == ord('n'):
			ith += 1
			ith = ith % 24

		if key == ord('w'):
			lpos[0][1] += 1
		if key == ord('s'):
			lpos[0][1] -= 1
		if key == ord('a'):
			lpos[0][0] -= 1
		if key == ord('d'):
			lpos[0][0] += 1

		if key == ord('f'):
			lpos[0][2] -= 1
		if key == ord('r'):
			lpos[0][2] += 1

		if key == ord('g'):
			lint -= 0.01
		if key == ord('t'):
			lint += 0.01

		if key == ord('h'):
			gamma -= 0.1
			svbrdf = {'diffuse': pow(d, gamma),'specular': s,'glossiness': g,'normal': n}
		if key == ord('y'):
			gamma += 0.1
			svbrdf = {'diffuse': pow(d, gamma),'specular': s,'glossiness': g,'normal': n}

		print("light pos: {}   light int: {:.4f}  gamma: {:.4f}\r".format(lpos, lint, gamma), end="")

		lightPos = tf.constant(lpos, name='DefaultLightPosition', dtype=tf.float32)
		lightInt = tf.constant([[lint, lint, lint]], name='DefaultLightIntensity', dtype=tf.float32)
		cameraPos = tf.constant(campos, name='DefaultCameraPosition', dtype=tf.float32)
		pred_images = render.render(pred_svbrdf, lightPos=lightPos, lightInt=lightInt, cameraPos=cameraPos)
		# print('pred_svbrdf', pred_svbrdf['diffuse'].shape)
		# print('pred_images', pred_images.shape)
		# cv2.imshow('prediction', cv2.cvtColor(pred_images[:, :, 0, :].numpy(), cv2.COLOR_RGB2BGR))
		cv2.imshow('prediction', cv2.cvtColor(pred_images[0, :, :, 0, :].numpy(), cv2.COLOR_RGB2BGR))

		# print(pred_svbrdf['normal'][0].numpy())
		cv2.imshow('diffuse', cv2.cvtColor(pred_svbrdf['diffuse'][0].numpy(), cv2.COLOR_RGB2BGR))
		cv2.imshow('specular', cv2.cvtColor(pred_svbrdf['specular'][0].numpy(), cv2.COLOR_RGB2BGR))
		cv2.imshow('normal', cv2.cvtColor(pred_svbrdf['normal'][0].numpy() * 0.5 + 0.5, cv2.COLOR_RGB2BGR))
		cv2.imshow('glossiness', pred_svbrdf['glossiness'][0].numpy())
		# cv2.imshow('img24', cv2.cvtColor(img24[..., ith,:].numpy(), cv2.COLOR_RGB2BGR))

		# print(src.shape)
		# if ds.include_src:
		# 	cv2.imshow('src', cv2.cvtColor(src[..., ith,:].numpy(), cv2.COLOR_RGB2BGR))

		key = cv2.waitKey(0)
	print('\r')