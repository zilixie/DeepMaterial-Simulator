import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import math
import cv2
import numpy as np
from svbrdf import *
import tensorflow as tf


def getPositionMap(size, fsizeMM):
	import numpy
	image = numpy.ndarray((size, size, 3), numpy.float32)
	hsize = fsizeMM * 0.5
	scale = fsizeMM / size
	for y in range(0, size):
		for x in range(0, size):
			fx = (x + 0.5) * scale - hsize
			fy = hsize - (y + 0.5) * scale
			image[y, x] = (fx, fy, 0)
	return image


class Render:
	def __init__(self, tex_res=512, tex_scale=100):
		lightPos = []
		lightInt = []

		R = 225
		alpha = [67.5, 45, 22.5]
		beta = [45, 90, 135, 180, 225, 270, 315, 0] 
		# beta = [180, 135, 90, 45, 0, 315, 270, 225]
		for a in alpha:
			z = R * math.sin(math.radians(a))
			r = R * math.cos(math.radians(a))
			for b in beta:
				x = r * math.cos(math.radians(b))
				y = r * math.sin(math.radians(b))
				lightPos.append([x,y,z])
				lightInt.append([0.18, 0.18, 0.18])

		# print(lightPos)
		self.lightPos = tf.constant(lightPos, name='DefaultLightPosition', dtype=tf.float32)
		self.lightInt = tf.constant(lightInt, name='DefaultLightIntensity', dtype=tf.float32)

		# default camera position
		self.cameraPos = tf.constant([0, 0, 225], name='DefaultCameraPosition', dtype=tf.float32)
		# pixel position
		self.positionMap = getPositionMap(tex_res, tex_scale)


	def render(self, svbrdf, lightPos=None, lightInt=None, cameraPos=None, positionMap=None, unflatten=True):  
		if lightPos is None: 
			lightPos = self.lightPos
		if lightInt is None: 
			lightInt = self.lightInt
		if cameraPos is None: 
			cameraPos = self.cameraPos
		if positionMap is None: 
			positionMap = self.positionMap

		position = tf.expand_dims(positionMap, -2)
		V = tf.math.l2_normalize(cameraPos - position, axis=-1)

		L, D = tf.linalg.normalize(lightPos - position, axis=-1)
		radiance = lightInt / (D * D / 1e6)

		d = pow(svbrdf['diffuse'], 1.2)
		s = pow(svbrdf['specular'], 2.2)
		g = svbrdf['glossiness']
		n = svbrdf['normal']
		diffuse = tf.expand_dims(d, -2)
		specular = tf.expand_dims(s, -2)
		glossiness = tf.expand_dims(g, -2)
		normal = tf.expand_dims(n, -2)

		image = ShadePBRSpecular(diffuse, specular, glossiness, normal, L, V) * radiance
		if unflatten:
			return tf.clip_by_value(image, 0.0, 1.0)

		image = tf.reshape(image, list(image.shape)[:-2] + [-1])
		return tf.clip_by_value(image, 0.0, 1.0)


if __name__ == '__main__':
	pass
	
	# render = Render()

	# key = None
	# ith = 0

	# ds1 = Dataset2('../pbr/pbr', include_src=False, dstype=0)
	# ds2 = Dataset2('../1911', include_src=True, dstype=1)
	# ds = [ds1, ds2]

	# d,s,n,g, src = ds[0].nxt()
	# img24 = render.render({'diffuse': d, 'specular': s, 'glossiness': g, 'normal': n})
	# # print(img24.shape)

	# switch = 0
	# while key != ord('z'):
	# 	if key == ord(' '):
			
	# 		if not switch:
	# 			d,s,n,g, src = ds[0].nxt()
	# 			img24 = render.render({'diffuse': d, 'specular': s, 'glossiness': g, 'normal': n})
	# 			# switch = 1
	# 		else:
	# 			d,s,n,g, src = ds[1].nxt()
	# 			img24 = src
	# 			# switch = 0

	# 		# d,s,n,g, src = ds1.nxt()
	# 		# img24 = render.render({'diffuse': d, 'specular': s, 'glossiness': g, 'normal': n})


	# 	if key == ord('n'):
	# 		ith += 1
	# 		ith = ith % 24

	# 	cv2.imshow('diffuse', cv2.cvtColor(d.numpy(), cv2.COLOR_RGB2BGR))
	# 	cv2.imshow('specular', cv2.cvtColor(s.numpy(), cv2.COLOR_RGB2BGR))
	# 	cv2.imshow('normal', cv2.cvtColor(n.numpy() * 0.5 + 0.5, cv2.COLOR_RGB2BGR))
	# 	cv2.imshow('glossiness', g.numpy())
	# 	cv2.imshow('renderred', cv2.cvtColor(img24[..., ith,:].numpy(), cv2.COLOR_RGB2BGR))

	# 	key = cv2.waitKey(0)
