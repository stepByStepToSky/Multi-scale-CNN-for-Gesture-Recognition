#!/usr/bin/env python2
# -*- coding: utf-8 -*-



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as framework

import deform_util


'''
only deformable

def cnn_model(x, x_left, x_right, batch_size, is_train):
    net = tf.reshape(x, [-1, 11, 39, 3])
    
    
    net, offset_map = deform_util.deform_conv2d(net, [5,5,3,18], [3,3,3,48], batch_size, activation=tf.nn.relu, scope="deform_conv1")
    net = layers.max_pool2d(net, [1,3], stride = [1,2], scope = 'CNN_maxpool1')
    print(net.shape)
    
    
    net = layers.convolution2d(net, 64, [3,5], padding = 'SAME', scope = 'CNN_conv2')
    net = layers.max_pool2d(net, [1,3], stride = [1,2], scope = 'CNN_maxpool2')
    print(net.shape)
    
    #net = layers.convolution2d(net, 32, [2,5], padding = 'VALID', scope = 'CNN_conv20')
    
    net = layers.convolution2d(net, 64, [3,3], padding = 'SAME', scope = 'CNN_conv3')
    net = layers.max_pool2d(net, [1,3], stride = 2, scope = 'CNN_maxpool3')
    print(net.shape)
    
    
    net = layers.flatten(net, scope = 'CNN_flatten')
    net = layers.fully_connected(net, 256, scope = 'CNN_fully_connect1')
    net = layers.dropout(net, keep_prob = 0.5, is_training = is_train, scope = 'CNN_dropout')
    net = layers.fully_connected(net, 20, scope = 'CNN_fully_connect2')
    return net, offset_map

'''



'''
deformable and multi scale

def cnn_model(x, x_left, x_right, batch_size, is_train):
    net = tf.reshape(x, [-1, 11, 39, 3])
    
    
    net, offset_map = deform_util.deform_conv2d(net, [5,5,3,18], [3,3,3,48], batch_size, activation=tf.nn.relu, scope="deform_conv1")
    #net = layers.max_pool2d(net, [1,3], stride = [1,2], scope = 'CNN_maxpool1')
    print(net.shape)
    
    
    net_3 = layers.convolution2d(net, 32, [3,3], padding = 'SAME', scope = 'CNN_conv2_3')
    net_3 = layers.max_pool2d(net_3, [1,3], stride = [1,2], scope = 'CNN_maxpool2_3')
    
    net_5 = layers.convolution2d(net, 32, [3,5], padding = 'SAME', scope = 'CNN_conv2_5')
    net_5 = layers.max_pool2d(net_5, [1,3], stride = [1,2], scope = 'CNN_maxpool2_5')
    
    net_9 = layers.convolution2d(net, 32, [3,7], padding = 'SAME', scope = 'CNN_conv2_9')
    net_9 = layers.max_pool2d(net_9, [1,3], stride = [1,2], scope = 'CNN_maxpool2_9')
    
    net = tf.concat([net_3, net_5, net_9], 3)
    print(net.shape)
    
    #net = layers.convolution2d(net, 32, [2,5], padding = 'VALID', scope = 'CNN_conv20')
    
    net = layers.convolution2d(net, 64, [3,3], padding = 'SAME', scope = 'CNN_conv3')
    net = layers.max_pool2d(net, [1,3], stride = 2, scope = 'CNN_maxpool3')
    print(net.shape)
    
    
    net = layers.flatten(net, scope = 'CNN_flatten')
    net = layers.fully_connected(net, 256, scope = 'CNN_fully_connect1')
    net = layers.dropout(net, keep_prob = 0.5, is_training = is_train, scope = 'CNN_dropout')
    net = layers.fully_connected(net, 20, scope = 'CNN_fully_connect2')
    return net, offset_map
'''


'''
only multi scale
def cnn_model(x, x_left, x_right, batch_size, is_train):
    net = tf.reshape(x, [-1, 11, 39, 3])
    
    
    #net, offset_map = deform_util.deform_conv2d(net, [5,5,3,18], [3,3,3,48], batch_size, activation=tf.nn.relu, scope="deform_conv1")
    #net = layers.max_pool2d(net, [1,3], stride = [1,2], scope = 'CNN_maxpool1')
    #print(net.shape)
    
    
    net_3 = layers.convolution2d(net, 32, [3,3], padding = 'SAME', scope = 'CNN_conv2_3')
    net_3 = layers.max_pool2d(net_3, [1,3], stride = [1,2], scope = 'CNN_maxpool2_3')
    
    net_5 = layers.convolution2d(net, 32, [3,5], padding = 'SAME', scope = 'CNN_conv2_5')
    net_5 = layers.max_pool2d(net_5, [1,3], stride = [1,2], scope = 'CNN_maxpool2_5')
    
    net_9 = layers.convolution2d(net, 32, [3,7], padding = 'SAME', scope = 'CNN_conv2_9')
    net_9 = layers.max_pool2d(net_9, [1,3], stride = [1,2], scope = 'CNN_maxpool2_9')
    
    net = tf.concat([net_3, net_5, net_9], 3)
    print(net.shape)
    
    #net = layers.convolution2d(net, 32, [2,5], padding = 'VALID', scope = 'CNN_conv20')
    
    net = layers.convolution2d(net, 64, [3,3], padding = 'SAME', scope = 'CNN_conv3')
    net = layers.max_pool2d(net, [1,3], stride = 2, scope = 'CNN_maxpool3')
    print(net.shape)
    
    
    net = layers.flatten(net, scope = 'CNN_flatten')
    net = layers.fully_connected(net, 256, scope = 'CNN_fully_connect1')
    net = layers.dropout(net, keep_prob = 0.5, is_training = is_train, scope = 'CNN_dropout')
    net = layers.fully_connected(net, 20, scope = 'CNN_fully_connect2')
    return net      #, offset_map
    
'''


'''
deformable and multi scale


def cnn_model(x, x_left, x_right, batch_size, is_train):
    net = tf.reshape(x, [-1, 11, 39, 3])
    
    net_left = tf.reshape(x_left, [-1, 4, 39, 3])
    net_right = tf.reshape(x_right, [-1, 4, 39, 3])

    net, offset_map = deform_util.deform_conv2d(net, [5,5,3,18], [3,3,3,48], batch_size, activation=tf.nn.relu, scope="deform_conv1")
    #net = layers.max_pool2d(net, [1,3], stride = [1,2], scope = 'CNN_maxpool1')
    print(net.shape)
    
    
    net_3 = layers.convolution2d(net, 32, [3,3], padding = 'SAME', scope = 'CNN_conv2_3')
    net_3 = layers.max_pool2d(net_3, [1,3], stride = [1,2], scope = 'CNN_maxpool2_3')
    
    net_5 = layers.convolution2d(net, 32, [3,5], padding = 'SAME', scope = 'CNN_conv2_5')
    net_5 = layers.max_pool2d(net_5, [1,3], stride = [1,2], scope = 'CNN_maxpool2_5')
    
    net_9 = layers.convolution2d(net, 32, [3,7], padding = 'SAME', scope = 'CNN_conv2_9')
    net_9 = layers.max_pool2d(net_9, [1,3], stride = [1,2], scope = 'CNN_maxpool2_9')
    
    net = tf.concat([net_3, net_5, net_9], 3)
    print(net.shape)
    
    #net = layers.convolution2d(net, 32, [2,5], padding = 'VALID', scope = 'CNN_conv20')
    
    net = layers.convolution2d(net, 48, [3,3], padding = 'SAME', scope = 'CNN_conv3')
    net = layers.max_pool2d(net, [1,3], stride = 2, scope = 'CNN_maxpool3')
    print(net.shape)
    
    #add for spatial attention
     
    net_left = layers.convolution2d(net_left, 32, [3,9], padding = 'SAME', scope = 'CNN_conv_left1')
    net_right = layers.convolution2d(net_right, 32, [3,9], padding = 'SAME', scope = 'CNN_conv_right1')
    net_left = layers.convolution2d(net_left, 32, [2,5], padding = 'SAME', scope = 'CNN_conv2_left')
    net_right = layers.convolution2d(net_right, 32, [2,5], padding = 'SAME', scope = 'CNN_conv2_right')
    net_left = layers.flatten(net_left, scope = 'CNN_flatten_left')
    net_right = layers.flatten(net_right, scope = 'CNN_flatten_right')
    net_left = layers.fully_connected(net_left, 1536, scope = 'CNN_fully_connect1_left')
    net_left = layers.dropout(net_left, keep_prob = 0.3, is_training = is_train, scope = 'CNN_dropout_left')
    net_right = layers.fully_connected(net_right, 1536, scope = 'CNN_fully_connect1_right')
    net_right = layers.dropout(net_right, keep_prob = 0.3, is_training = is_train, scope = 'CNN_dropout_right')
    
    net = layers.flatten(net, scope = 'CNN_flatten')
    net = tf.concat([net, net_left, net_right], 1)

    net = layers.flatten(net, scope = 'CNN_flatten')
    net = layers.fully_connected(net, 256, scope = 'CNN_fully_connect1')
    net = layers.dropout(net, keep_prob = 0.5, is_training = is_train, scope = 'CNN_dropout')
    net = layers.fully_connected(net, 20, scope = 'CNN_fully_connect2')
    return net, offset_map
'''


'''not jointly train
'''

def cnn_model_hands(x_left, x_right, batch_size, is_train):
    net_left = tf.reshape(x_left, [-1, 4, 39, 3])
    net_right = tf.reshape(x_right, [-1, 4, 39, 3])
    #add for spatial attention left
    net_3_left = layers.convolution2d(net_left, 32, [3,3], padding = 'SAME', scope = 'l_CNN_conv2_3')
    net_3_left = layers.max_pool2d(net_3_left, [1,3], stride = [1,2], scope = 'l_CNN_maxpool2_3')
    net_5_left = layers.convolution2d(net_left, 32, [3,5], padding = 'SAME', scope = 'l_CNN_conv2_5')
    net_5_left = layers.max_pool2d(net_5_left, [1,3], stride = [1,2], scope = 'l_CNN_maxpool2_5')
    net_9_left = layers.convolution2d(net_left, 32, [3,7], padding = 'SAME', scope = 'l_CNN_conv2_9')
    net_9_left = layers.max_pool2d(net_9_left, [1,3], stride = [1,2], scope = 'l_CNN_maxpool2_9')
    net_left = tf.concat([net_3_left, net_5_left, net_9_left], 3)
    print(net_left.shape)
    net_left = layers.convolution2d(net_left, 48, [3,3], padding = 'SAME', scope = 'l_CNN_conv3')
    net_left = layers.max_pool2d(net_left, [1,3], stride = 2, scope = 'l_CNN_maxpool3')
    print(net_left.shape)
    net_left = layers.flatten(net_left, scope = 'l_CNN_flatten')
    net_left = layers.fully_connected(net_left, 512, scope = 'l_CNN_fully_connect1')
    net_left = layers.dropout(net_left, keep_prob = 0.5, is_training = is_train, scope = 'l_CNN_dropout')
    
    #add for spatial attention right
    net_3_right = layers.convolution2d(net_right, 32, [3,3], padding = 'SAME', scope = 'r_CNN_conv2_3')
    net_3_right = layers.max_pool2d(net_3_right, [1,3], stride = [1,2], scope = 'r_CNN_maxpool2_3')
    net_5_right = layers.convolution2d(net_right, 32, [3,5], padding = 'SAME', scope = 'r_CNN_conv2_5')
    net_5_right = layers.max_pool2d(net_5_right, [1,3], stride = [1,2], scope = 'r_CNN_maxpool2_5')
    net_9_right = layers.convolution2d(net_right, 32, [3,7], padding = 'SAME', scope = 'r_CNN_conv2_9')
    net_9_right = layers.max_pool2d(net_9_right, [1,3], stride = [1,2], scope = 'r_CNN_maxpool2_9')
    net_right = tf.concat([net_3_right, net_5_right, net_9_right], 3)
    print(net_right.shape)
    net_right = layers.convolution2d(net_right, 48, [3,3], padding = 'SAME', scope = 'r_CNN_conv3')
    net_right = layers.max_pool2d(net_right, [1,3], stride = 2, scope = 'r_CNN_maxpool3')
    print(net_right.shape)
    net_right = layers.flatten(net_right, scope = 'r_CNN_flatten')
    net_right = layers.fully_connected(net_right, 512, scope = 'r_CNN_fully_connect1')
    net_right = layers.dropout(net_right, keep_prob = 0.5, is_training = is_train, scope = 'r_CNN_dropout')
    
    # independant classifier for two hands
    net_hands = tf.concat([net_left, net_right], 1)
    net_hands = layers.flatten(net_hands, scope = 'lr_CNN_flatten')
    net_hands = layers.fully_connected(net_hands, 256, scope = 'lr_CNN_fully_connect1')
    net_hands = layers.dropout(net_hands, keep_prob = 0.5, is_training = is_train, scope = 'lr_CNN_dropout')
    net_hands = layers.fully_connected(net_hands, 20, scope = 'lr_CNN_fully_connect2')
    return net_hands, net_left, net_right

def cnn_model(x, net_left, net_right, batch_size, is_train):
    net = tf.reshape(x, [-1, 11, 39, 3])

    net, offset_map = deform_util.deform_conv2d(net, [5,5,3,18], [3,3,3,48], batch_size, activation=tf.nn.relu, scope="deform_conv1")
    #net = layers.max_pool2d(net, [1,3], stride = [1,2], scope = 'CNN_maxpool1')
    print(net.shape)
    net_3 = layers.convolution2d(net, 32, [3,3], padding = 'SAME', scope = 'g_CNN_conv2_3')
    net_3 = layers.max_pool2d(net_3, [1,3], stride = [1,2], scope = 'g_CNN_maxpool2_3')
    net_5 = layers.convolution2d(net, 32, [3,5], padding = 'SAME', scope = 'g_CNN_conv2_5')
    net_5 = layers.max_pool2d(net_5, [1,3], stride = [1,2], scope = 'g_CNN_maxpool2_5')
    net_9 = layers.convolution2d(net, 32, [3,7], padding = 'SAME', scope = 'g_CNN_conv2_9')
    net_9 = layers.max_pool2d(net_9, [1,3], stride = [1,2], scope = 'g_CNN_maxpool2_9')
    net = tf.concat([net_3, net_5, net_9], 3)
    print(net.shape)
    net = layers.convolution2d(net, 48, [3,3], padding = 'SAME', scope = 'g_CNN_conv3')
    net = layers.max_pool2d(net, [1,3], stride = 2, scope = 'g_CNN_maxpool3')
    print(net.shape)
    net = layers.flatten(net, scope = 'g_CNN_flatten')    
    
    net = tf.concat([net, net_left, net_right], 1)
    net = layers.flatten(net, scope = 'g_CNN_flatten')
    net = layers.fully_connected(net, 256, scope = 'g_CNN_fully_connect1')
    net = layers.dropout(net, keep_prob = 0.5, is_training = is_train, scope = 'g_CNN_dropout')
    net = layers.fully_connected(net, 20, scope = 'g_CNN_fully_connect2')
    
    return net, offset_map
    