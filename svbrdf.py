import os
import numpy as np
import cv2
import math
import tensorflow as tf


def normalize(image):
    return tf.math.l2_normalize(image, axis=-1)

def dot(a, b):
    return tf.reduce_sum(tf.multiply(a, b), -1, keepdims=True)

def FresnelSchlick(dotVH, F0):
    f = F0 + (1.0 - F0) * tf.pow((1-dotVH), 5)
    return tf.clip_by_value(f, 0.0, 1.0)


def ShadePBR(C_diff, F0, a, N, L, V):
    # GGX
    H = normalize(V + L)

    dotVH = tf.maximum(dot(V, H), 1e-6)
    dotNL = tf.maximum(dot(N, L), 1e-6)
    dotNV = tf.maximum(dot(N, V), 1e-6)
    dotNH = tf.maximum(dot(N, H), 1e-6)

    # Fresnel Schlick
    F = F0 + (1-F0) * tf.pow((1-dotVH), 5)

    # D - Trowbridge-Reitz
    a2 = a * a # roughness * roughness
    # print('a2', a2.shape)
    denom = dotNH * dotNH * (a2 - 1.0) + 1.0
    D = a2 / (denom * denom * math.pi)
    # print('D', D.shape)
    # DTR

    # G - Smith Joint GGX
    denom1 = dotNL * tf.math.sqrt(dotNV * dotNV * (1 - a2) + a2)
    denom2 = dotNV * tf.math.sqrt(dotNL * dotNL * (1 - a2) + a2)
    Vis = 0.5 / (denom1 + denom2)
    spec = F * Vis * D
    # print('spec', spec.shape)

    # Lambert
    # diff = k_diff * Cdiff/pi
    # spec = k_spec * VDF?
    diff = (1-F) * C_diff * 0.31830988618379067154 # (1/math.pi)
    shade_pbr = (diff + spec) * dotNL
    # print('shade_pbr',shade_pbr.shape)
    return shade_pbr


def ShadePBRSpecular(diffuse, specular, glossiness, N, L, V):
    # diffuse
    C_diff = diffuse * \
        (1 - tf.reduce_max(specular, axis=-1, keepdims=True))
    F0 = specular
    a = tf.pow((1 - glossiness), 2)

    return ShadePBR(C_diff, F0, a, N, L, V)
