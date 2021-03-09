# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:31:49 2018

@author: ZJU
"""

import tensorflow as tf
import scipy.io as spio


# 加载训练好的vgg19中的参数
class Vgg19:
    
    def __init__(self, image_tensor):
        self.image_tensor = image_tensor
        self.create_layers()
        
        
    def create_conv(self, prev_layer, name, weights_as_list):
        with tf.variable_scope(name) as scope:
            weights = tf.constant(weights_as_list, name=name+'_weights')
            conv_layer = tf.nn.conv2d(prev_layer, weights, [1,1,1,1], 'SAME', name=name)
        
            return conv_layer
    
    
    def create_relu(self, prev_layer, name, biases_as_list):
        biases_as_list = biases_as_list.reshape(biases_as_list.size)
        with tf.variable_scope(name) as scope:
            biases = tf.constant(biases_as_list, name=name+'_biases')
            relu_layer = tf.nn.relu(prev_layer+biases, name=name)
            
            return relu_layer
        
        
    def create_pool(self, prev_layer, name, is_avg=True):
        pooling_type = 'MAX'
        if is_avg:
            pooling_type = 'AVG'
        with tf.variable_scope(name) as scope:
            pool_layer = tf.nn.pool(input=prev_layer, window_shape=[2,2], pooling_type=pooling_type, padding='SAME', strides=[2,2], data_format='NHWC', name=name)
            
            return pool_layer
            
        
    def create_layers(self):
        mat = spio.loadmat('../Internal-External/imagenet-vgg-verydeep-19.mat')
        layers = mat['layers']
        
        self.conv1_1 = self.create_conv(self.image_tensor, 'conv1_1', layers[0][0][0][0][2][0][0])
        self.relu1_1 = self.create_relu(self.conv1_1, 'relu1_1', layers[0][0][0][0][2][0][1])
        self.conv1_2 = self.create_conv(self.relu1_1, 'conv1_2', layers[0][2][0][0][2][0][0])
        self.relu1_2 = self.create_relu(self.conv1_2, 'relu1_2', layers[0][2][0][0][2][0][1])
        self.pool1 = self.create_pool(self.relu1_2, 'pool1')
        
        self.conv2_1 = self.create_conv(self.pool1, 'conv2_1', layers[0][5][0][0][2][0][0])
        self.relu2_1 = self.create_relu(self.conv2_1, 'relu2_1', layers[0][5][0][0][2][0][1])
        self.conv2_2 = self.create_conv(self.relu2_1, 'conv2_2', layers[0][7][0][0][2][0][0])
        self.relu2_2 = self.create_relu(self.conv2_2, 'relu2_2', layers[0][7][0][0][2][0][1])
        self.pool2 = self.create_pool(self.relu2_2, 'pool2')
        
        self.conv3_1 = self.create_conv(self.pool2, 'conv3_1', layers[0][10][0][0][2][0][0])
        self.relu3_1 = self.create_relu(self.conv3_1, 'relu3_1', layers[0][10][0][0][2][0][1])
        self.conv3_2 = self.create_conv(self.relu3_1, 'conv3_2', layers[0][12][0][0][2][0][0])
        self.relu3_2 = self.create_relu(self.conv3_2, 'relu3_2', layers[0][12][0][0][2][0][1])
        self.conv3_3 = self.create_conv(self.relu3_2, 'conv3_3', layers[0][14][0][0][2][0][0])
        self.relu3_3 = self.create_relu(self.conv3_3, 'relu3_3', layers[0][14][0][0][2][0][1])
        self.conv3_4 = self.create_conv(self.relu3_3, 'conv3_4', layers[0][16][0][0][2][0][0])
        self.relu3_4 = self.create_relu(self.conv3_4, 'relu3_4', layers[0][16][0][0][2][0][1])
        self.pool3 = self.create_pool(self.relu3_4, 'pool3')
        
        self.conv4_1 = self.create_conv(self.pool3, 'conv4_1', layers[0][19][0][0][2][0][0])
        self.relu4_1 = self.create_relu(self.conv4_1, 'relu4_1', layers[0][19][0][0][2][0][1])
        self.conv4_2 = self.create_conv(self.relu4_1, 'conv4_2', layers[0][21][0][0][2][0][0])
        self.relu4_2 = self.create_relu(self.conv4_2, 'relu4_2', layers[0][21][0][0][2][0][1])
        self.conv4_3 = self.create_conv(self.relu4_2, 'conv4_3', layers[0][23][0][0][2][0][0])
        self.relu4_3 = self.create_relu(self.conv4_3, 'relu4_3', layers[0][23][0][0][2][0][1])
        self.conv4_4 = self.create_conv(self.relu4_3, 'conv4_4', layers[0][25][0][0][2][0][0])
        self.relu4_4 = self.create_relu(self.conv4_4, 'relu4_4', layers[0][25][0][0][2][0][1])
        self.pool4 = self.create_pool(self.relu4_4, 'pool4')
        
        self.conv5_1 = self.create_conv(self.pool4, 'conv5_1', layers[0][28][0][0][2][0][0])
        self.relu5_1 = self.create_relu(self.conv5_1, 'relu5_1', layers[0][28][0][0][2][0][1])
        self.conv5_2 = self.create_conv(self.relu5_1, 'conv5_2', layers[0][30][0][0][2][0][0])
        self.relu5_2 = self.create_relu(self.conv5_2, 'relu5_2', layers[0][30][0][0][2][0][1])
        
        self.conv5_3 = self.create_conv(self.relu5_2, 'conv5_3', layers[0][32][0][0][2][0][0])
        self.relu5_3 = self.create_relu(self.conv5_3, 'relu5_3', layers[0][32][0][0][2][0][1])
        self.conv5_4 = self.create_conv(self.relu5_3, 'conv5_4', layers[0][34][0][0][2][0][0])
        self.relu5_4 = self.create_relu(self.conv5_4, 'relu5_4', layers[0][34][0][0][2][0][1])
        self.pool5 = self.create_pool(self.relu5_4, 'pool5')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    

