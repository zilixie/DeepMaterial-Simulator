import os, math, random, threading, queue, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import moderngl
import numpy as np
import tensorflow as tf

from svbrdf import *
from render import Render

from datetime import datetime
from network import SVBRDF_Net


cropped_imgs = []
seq = range(0, 24)

for i in seq:
	print(i)
	path = '../11/P_L{}E.png'
	img = cv2.cvtColor(cv2.imread(path.format(i)),  cv2.COLOR_RGB2BGR)

	cropped_img = tf.image.crop_to_bounding_box(img, 274, 786, 1500, 1500)
	cropped_img = tf.image.resize(cropped_img, [512, 512]).numpy()

	cropped_img = cropped_img/255.0
	cropped_imgs.append(cropped_img)

	

cropped_imgs = np.array(cropped_imgs)
cropped_imgs = np.expand_dims(np.transpose(cropped_imgs, (1,2,0,3)), 0)
print(cropped_imgs.shape)

cropped_imgs = tf.reshape(cropped_imgs, list(cropped_imgs.shape)[:-2] + [-1])
cropped_imgs = tf.clip_by_value(cropped_imgs, 0.0, 1.0)
model = tf.keras.models.load_model('../model_0390.h5')


pred = model(tf.concat(cropped_imgs, axis=-1), training=False)
pred_svbrdf = {'diffuse': pred[0],'specular': pred[1],'glossiness': pred[2],'normal': pred[3]}

key = None
while key != ord('z'):

	diffuse = cv2.cvtColor(pred_svbrdf['diffuse'][0].numpy(), cv2.COLOR_RGB2BGR)
	specular = cv2.cvtColor(pred_svbrdf['specular'][0].numpy(), cv2.COLOR_RGB2BGR)
	glossiness = pred_svbrdf['glossiness'][0].numpy()
	normal = cv2.cvtColor(pred_svbrdf['normal'][0].numpy() * 0.5 + 0.5, cv2.COLOR_RGB2BGR)

	cv2.imshow('diffuse', diffuse)
	cv2.imshow('specular', specular)
	cv2.imshow('glossiness', glossiness)
	cv2.imshow('normal', normal)
	key = cv2.waitKey(0)

cv2.imwrite('./result/diffuse.png', diffuse * 255.0)
cv2.imwrite('./result/specular.png', specular * 255.0)
cv2.imwrite('./result/glossiness.png', glossiness * 255.0)
cv2.imwrite('./result/normal.png', normal * 255.0)

