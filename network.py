import math, os, logging
import tensorflow as tf
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.get_logger().setLevel(tf.compat.v1.logging.FATAL)

def downsample(num_channels, idx):
    model = tf.keras.Sequential([
        layers.Conv2D(num_channels, 3, 2, 'same'),
        layers.LeakyReLU(),
        layers.Conv2D(num_channels, 3, 1, 'same'),
        layers.LeakyReLU(),
        layers.Conv2D(num_channels, 3, 1, 'same'),
        layers.LeakyReLU()
    ], name='downsample_%d'% idx)
    return model

def upsample(num_channels, idx):
    model = tf.keras.Sequential([
        #layers.UpSampling2D(interpolation='bilinear'),
        layers.Conv2DTranspose(num_channels, 3, 2, 'same'),
        layers.LeakyReLU(),
        layers.Conv2D(num_channels, 3, 1, 'same'),
        layers.LeakyReLU(),
        layers.Conv2D(num_channels, 3, 1, 'same'),
        layers.LeakyReLU()
    ], name='upsample_%d'% idx)
    return model


def SVBRDF_Net(image_res, num_images):
    num_channels, inc_channels = 64, 32
    num_levels = int(math.log2(image_res) - 3)

    # os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    # logging.getLogger('tensorflow').setLevel(logging.FATAL)
    # input shape: (image_res, image_res, num_images*3)
    inputs = tf.keras.Input(shape=(image_res, image_res, num_images * 3), name="images")
    
    # preprocessing, get (image_res, image_res, num_channels) feature map from input
    feat = layers.Conv2D(num_channels, 3, 1, 'same', activation=None)(inputs)
    feat = layers.LeakyReLU()(feat) 

    # down sampling
    convs = []
    for i in range(num_levels):
        num_channels += inc_channels
        feat = downsample(num_channels, i)(feat)
        convs.append(feat)

    # middle convs
    mid = tf.keras.Sequential([
        layers.Conv2D(num_channels, 3, 1, 'same', activation=None),
        layers.LeakyReLU(),
        layers.Conv2D(num_channels, 3, 1, 'same', activation=None),
        layers.LeakyReLU(),
        layers.Conv2D(num_channels, 3, 1, 'same', activation=None),
        layers.LeakyReLU(),
        layers.Conv2D(num_channels, 3, 1, 'same', activation=None),
        layers.LeakyReLU(),
    ] * 4, name='middle')
    
    feat = mid(feat)

    # up sampling
    for i in range(num_levels):
        num_channels -= inc_channels
        #feat = layers.Concatenate()([feat, convs[-(i+1)]])
        feat = layers.Add()([feat, convs[-(i+1)]])
        feat = upsample(num_channels, i)(feat)

    # post processing, get (image_res, image_res, num_channels) feature map
    feat = layers.Conv2D(num_channels, 3, 1, 'same', activation=None)(feat)
    feat = layers.LeakyReLU()(feat)

    # output        
    diffuse =  layers.Conv2D(3, 1, 1, 'same', activation='sigmoid', name='diffuse')(feat)
    specular = layers.Conv2D(3, 1, 1, 'same', activation='sigmoid', name='specular')(feat)
    glossiness = layers.Conv2D(1, 1, 1, 'same', activation='sigmoid', name='glossiness')(feat)
    normal =  layers.Conv2D(3, 1, 1, 'same', activation='tanh', name='normal')(feat)

    # normalize data
    glossiness = layers.Lambda(lambda tensor: tf.keras.backend.clip(tensor, 0.01, 0.99), name='n_glossiness')(glossiness)
    normal = layers.Lambda(lambda tensor: tf.keras.backend.l2_normalize(tensor, axis=-1), name='n_normal')(normal)
    
    return tf.keras.Model(inputs=inputs, outputs=[diffuse, specular, glossiness, normal])