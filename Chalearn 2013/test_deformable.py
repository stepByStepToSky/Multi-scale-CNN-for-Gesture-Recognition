
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as framework

import numpy
import os
import model_deformable

isRestoreModel = True
restore_model_path = os.getcwd() + "/model/"

def read_hdf5(path):
    with h5py.File(path, 'r') as hf:
        print('Read matrix and label from ', path)
        matrix = np.array(hf.get('matrix'))
        label = np.array(hf.get('label'))
        print("Load matrix shape is ", matrix.shape, ' Load label shape is ', label.shape)
        return matrix, label

def test():
    path_val = os.path.join(os.getcwd(), "val.h5")
    
    x_val, y_val = read_hdf5(path_val)
    x_val = x_val[:, :11, :, :]
    x_val_left = x_val[:, 3:7, :, :]
    x_val_right = x_val[:, 7:11, :, :]
    
    is_training = tf.placeholder(tf.bool)
    batch_size_place = tf.placeholder(tf.int32)
    x_place = tf.placeholder(tf.float32, [None, 11, 39, 3])
    x_left_place = tf.placeholder(tf.float32, [None, 4, 39, 3])
    x_right_place = tf.placeholder(tf.float32, [None, 4, 39, 3])
    y_place = tf.placeholder(tf.float32, [None, 20])
    
    net_hands, net_left, net_right = model_deformable.cnn_model_hands(x_left_place, x_right_place, batch_size_place, is_training)
    y, offset_map = model_deformable.cnn_model(x_place, net_left, net_right, batch_size_place, is_training)
    
    #offset_map_h = tf.reshape(offset_map[...,0], [batch_size_place, 11, 39, 5, 5])
    #offset_map_w = tf.reshape(offset_map[...,1], [batch_size_place, 11, 39, 5, 5])
    
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.argmax(y_place, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer(), feed_dict = {is_training: True})
    
    # restore
    if isRestoreModel:
        saver.restore(sess, tf.train.latest_checkpoint(restore_model_path))
        print('restore model from %s' %(tf.train.latest_checkpoint(restore_model_path)))
    
              
    val_acc = sess.run(accuracy, feed_dict={batch_size_place:3434, is_training:False, x_place:x_val, \
                                                    x_left_place:x_val_left, x_right_place:x_val_right, y_place:y_val})
    
    print('val_acc %.5f' %val_acc)
            
if __name__ == '__main__':
    test()
