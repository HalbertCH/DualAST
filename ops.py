# Copyright (C) 2018  Artsiom Sanakoyeu and Dmytro Kotovenko
#
# This file is part of Adaptive Style Transfer
#
# Adaptive Style Transfer is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Adaptive Style Transfer is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib as tf_contrib
from tensorflow.python.framework import ops
import cv2

import tensorflow.contrib.layers as tflayers

from utils import *


factor, mode, uniform = pytorch_kaiming_weight_factor(a=0.0, uniform=False)
weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)


def batch_norm(input, is_training=True, name="batch_norm"):
    x = tflayers.batch_norm(inputs=input,
                            scale=True,
                            is_training=is_training,
                            trainable=True,
                            reuse=None)
    return x


def instance_norm(input, name="instance_norm", is_training=True):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset


def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d", activation_fn=None):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=activation_fn,
                           weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                           biases_initializer=None)


def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    # Upsampling procedure, like suggested in this article:
    # https://distill.pub/2016/deconv-checkerboard/. At first upsample
    # tensor like an image and then apply convolutions.
    with tf.variable_scope(name):
        input_ = tf.image.resize_images(images=input_,
                                        size=tf.shape(input_)[1:3] * s,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # That is optional
        return conv2d(input_=input_, output_dim=output_dim, ks=ks, s=1, padding='SAME')


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def fully_connected(x, units, use_bias=True, scope='fully_connected'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                            kernel_regularizer=weight_regularizer,
                            use_bias=use_bias)

        return x


def flatten(x) :
    return tf.layers.flatten(x)


def MLP(style, channel, n_res, scope='MLP'):
    with tf.variable_scope(scope):
        x = style

        for i in range(2):
            x = fully_connected(x, channel, scope='FC_' + str(i))
            x = tf.nn.relu(x)

        mu_list = []
        var_list = []

        for i in range(n_res * 2):
            mu = fully_connected(x, channel, scope='FC_mu_' + str(i))
            var = fully_connected(x, channel, scope='FC_var_' + str(i))

            mu = tf.reshape(mu, shape=[-1, 1, 1, channel])
            var = tf.reshape(var, shape=[-1, 1, 1, channel])

            mu_list.append(mu)
            var_list.append(var)

        return mu_list, var_list


def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP
    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)
    return gamma * ((content - c_mean) / c_std) + beta


def adaptive_avg_pooling(x):
    # global average pooling
    gap = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    return gap

