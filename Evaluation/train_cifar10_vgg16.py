# -*- coding:utf-8 -*-  
import tensorflow as tf

def bias_variable(shape,name,trainable=True):
    initial = tf.constant(0., shape=shape)
    return tf.get_variable(name,initializer=initial,trainable=trainable)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(input, k_size=1, stride=1, name=None):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1],
                          padding='SAME', name=name)


def batch_norm(input, train_flag):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)


def Vgg_16(x,  keep_prob=None,train_flag=None,reuse=None):
    # build_network
    with tf.variable_scope('VGG16',reuse=reuse):
  
        W_conv1_1 = tf.get_variable('w_conv1_1', shape=[3, 3, 1, 64], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv1_1 = bias_variable([64],name='b_conv1_1')
        output = tf.nn.relu(batch_norm(conv2d(x, W_conv1_1) + b_conv1_1, train_flag))
    
    
        W_conv1_2 = tf.get_variable('w_conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv1_2 = bias_variable([64],name='b_conv1_2')
        output = tf.nn.relu(batch_norm(conv2d(output, W_conv1_2) + b_conv1_2, train_flag))
        output = max_pool(output, 2, 2, "pool1")
    
    
        W_conv2_1 = tf.get_variable('w_conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv2_1 = bias_variable([128],name='b_conv2_1')
        output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_1) + b_conv2_1, train_flag))
    
    
        W_conv2_2 = tf.get_variable('w_conv2_2', shape=[3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv2_2 = bias_variable([128],name='b_conv2_2')
        output = tf.nn.relu(batch_norm(conv2d(output, W_conv2_2) + b_conv2_2, train_flag))
        output = max_pool(output, 2, 2, "pool2")
    
    
        W_conv3_1 = tf.get_variable('w_conv3_1', shape=[3, 3, 128, 256], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3_1 = bias_variable([256],name='b_conv3_1')
        output = tf.nn.relu( batch_norm(conv2d(output,W_conv3_1) + b_conv3_1, train_flag))
    
    
        W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3_2 = bias_variable([256],name='b_conv3_2')
        output = tf.nn.relu(batch_norm(conv2d(output, W_conv3_2) + b_conv3_2, train_flag))
    
    
        W_conv3_3 = tf.get_variable('w_conv3_3', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
        b_conv3_3 = bias_variable([256],name='b_conv3_3')
        output = tf.nn.relu( batch_norm(conv2d(output, W_conv3_3) + b_conv3_3, train_flag))
        output = max_pool(output, 2, 2, "pool3")

        output = tf.reshape(output, [-1, 28*44*256])
    
        return output

