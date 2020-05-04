# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import math
import glob
from PIL import Image
from scipy.signal import convolve2d

batch_size = 1
Block_size = 32
image_channel = 1
PhaseNumber = 10
mode = 'test'
rate_list = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4]
k = len(rate_list)

Matrix = 'D:/DeepBCS-master/Train/Pre-train/phi_0.1_7'  
Phi_matrix = sio.loadmat(Matrix)
A1 = Phi_matrix['A1']
A2 = Phi_matrix['A2']
A3 = Phi_matrix['A3']
A4 = Phi_matrix['A4']
A5 = Phi_matrix['A5']
A6 = Phi_matrix['A6']
A7 = Phi_matrix['A7']

Pinv = 'D:/DeepBCS-master/Train/Pre-train/phi_pinv_0.1_7'
Phi_pinv = sio.loadmat(Pinv)
A1_pinv = Phi_pinv['A1_pinv']
A2_pinv = Phi_pinv['A2_pinv']
A3_pinv = Phi_pinv['A3_pinv']
A4_pinv = Phi_pinv['A4_pinv']
A5_pinv = Phi_pinv['A5_pinv']
A6_pinv = Phi_pinv['A6_pinv']
A7_pinv = Phi_pinv['A7_pinv']


if mode == 'test':
    rate = 0.1  #the adaptive sampling rate {0.03, 0.05, 0.1, 0.2, 0.3}   
    #for Set5
    filename='baby'  # for Set5{baby, bird, butterfly, head, woman}
    testdata_path = 'D:/DeepBCS-master/DataSets/TestData/Set5' 
    filepaths = glob.glob(testdata_path + '\%s.bmp' %filename)  
    test_onehot_path = 'D:/DeepBCS-master/Test/Onehot/Set5/%s_7_3/onehot_%s_%s'  %(rate, filename, rate)

    #for Set11 or BSD100
    '''
    testdata_path = 'D:/DeepBCS-master/DataSets/TestData/Set11_256' 
    filepaths = glob.glob(testdata_path + '/*.tif') 
    test_onehot_path = 'D:/DeepBCS-master/Test/Onehot/Set11/%s_7_3/onehot_Set11_256_%s' %(rate, rate)
    '''
    
    image_size_row = 512  #the size of test image
    image_size_col = 512
    nrtest = len(filepaths)
    Block_num_m = np.int(image_size_row / Block_size)
    Block_num_n = np.int(image_size_col / Block_size)  
    
    save_paths = 'D:/DeepBCS-master/DataSets/Prediction/%s/' %rate
    if os.path.exists(save_paths) == 0:
        os.makedirs(save_paths)
        
    Onehot_name = sio.loadmat(test_onehot_path)
    Block_onehot = np.float32(Onehot_name['Block_onehot'])    
    if nrtest == 1:
        Block_onehot = Block_onehot[np.newaxis, :, :]
    else:
        Block_onehot = np.transpose(Block_onehot, (2, 0, 1))


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def WeightsVariable(shape, name='w', trainable=True):
    if len(shape) == 4:
        N = shape[0] * shape[1] * (shape[2] + shape[3]) / 2
    else:
        N = shape[0] / 2
    initial = tf.random_normal(shape=shape, stddev=np.sqrt(2.0 / N)) 
    return tf.get_variable(name, initializer=initial, trainable=trainable)


def add_con2d_weight_bias(w_shape, b_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                              name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]


def Conv2d(x, w, b, stride=1, padding='SAME'):
    return tf.nn.bias_add(tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding), b)


def Conv2d_no_bias(x, w, stride=1, padding='SAME'):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)


def ista_block(input_layers, Initial_PhiTY, Initial_PhiTPhi, layer_no):
    num = 0
    PhiTPhiX = []

    for p in range(Block_num_m):  
        PhiTPhiX_row = []
        for q in range(Block_num_n):
            image_block = input_layers[-1][:, p * Block_size:(p + 1) * Block_size, q * Block_size:(q + 1) * Block_size,
                          :]  
            image_block = tf.reshape(image_block, [Block_size * Block_size, -1])  
            PhiTPhi_block = tf.reshape(Initial_PhiTPhi[num],
                                       [Block_size * Block_size, Block_size * Block_size]) 
            PhiTPhiX_block = tf.matmul(PhiTPhi_block, image_block) 
            PhiTPhiX_block = tf.reshape(PhiTPhiX_block, [-1, Block_size, Block_size, 1]) 
            num = num + 1
            if q == 0:
                PhiTPhiX_row = PhiTPhiX_block
            else:
                PhiTPhiX_row = tf.concat([PhiTPhiX_row, PhiTPhiX_block], 2)
        if p == 0:
            PhiTPhiX = PhiTPhiX_row
        else:
            PhiTPhiX = tf.concat([PhiTPhiX, PhiTPhiX_row], 1)

    conv_size = 64
    filter_size = 3
    [Weights1, bias1] = add_con2d_weight_bias([filter_size, filter_size, 1, conv_size], [conv_size], 1)
    [Weights2, bias2] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 2)
    [Weights3, bias3] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 3)
    # [Weights4, bias4] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 4)
    [Weights5, bias5] = add_con2d_weight_bias([filter_size, filter_size, conv_size, 1], [1], k)

    r_Deep = input_layers[-1] + Initial_PhiTY - PhiTPhiX
    x1_deep = tf.nn.relu(Conv2d(r_Deep, Weights1, bias1, stride=1, padding='SAME'))
    x2_deep = tf.nn.relu(Conv2d(x1_deep, Weights2, bias2, stride=1, padding='SAME'))
    x3_deep = tf.nn.relu(Conv2d(x2_deep, Weights3, bias3, stride=1, padding='SAME'))
    # x4_deep = tf.nn.relu(Conv2d(x3_deep, Weights4, bias4, stride=1, padding='SAME'))
    x5_deep = Conv2d(x3_deep, Weights5, bias5, stride=1, padding='SAME')

    x6_deep = x5_deep + r_Deep  
    return x6_deep


def Deep_Net(Initial_image, Initial_PhiTY, Initial_PhiTPhi, n):
    layers = []
    layers.append(Initial_image)
    for i in range(n):
        with tf.variable_scope('deep_conv_%d' % i, reuse=tf.AUTO_REUSE):
            Deep = ista_block(layers, Initial_PhiTY, Initial_PhiTPhi, i)
            layers.append(Deep)
    Y_pred = Initial_image + layers[-1]
    Y_medium = layers[-5]
    return Y_pred, Y_medium


def Initial_Net(Block_image, Block_sampling_rate, Phi_A, Phi_A_pinv):
    nB = np.int32(np.round(Block_size * Block_size * Block_sampling_rate))
    with tf.variable_scope('sample_initial_' + str(Block_sampling_rate) + str(nB), reuse=tf.AUTO_REUSE):
        weight1 = WeightsVariable(shape=[Block_size, Block_size, image_channel, nB], name='w1')
        conv1 = Conv2d_no_bias(Block_image, weight1, stride=Block_size, padding='VALID')
        Block_y = tf.reshape(tf.squeeze(conv1), [nB, 1])
        PhiTY = tf.matmul(Phi_A_pinv, Block_y) 
        PhiTPhi = tf.matmul(Phi_A_pinv, Phi_A)  

        weight2 = WeightsVariable(shape=[1, 1, nB, Block_size * Block_size], name='w2')
        conv2 = Conv2d_no_bias(conv1, weight2, stride=1, padding='VALID')
        y_Block_pred = tf.reshape(tf.squeeze(conv2), [-1, Block_size, Block_size, image_channel])
    return [y_Block_pred, PhiTY, PhiTPhi]


with tf.Graph().as_default():
    # 构造图
    x_image = tf.placeholder(tf.float32, [batch_size, image_size_row, image_size_col, image_channel])
    Onehot = tf.placeholder(tf.float32, [batch_size, Block_onehot.shape[1], k])
    Phi_A1 = tf.placeholder(tf.float32, [A1.shape[0], A1.shape[1]])
    Phi_A2 = tf.placeholder(tf.float32, [A2.shape[0], A2.shape[1]])
    Phi_A3 = tf.placeholder(tf.float32, [A3.shape[0], A3.shape[1]])
    Phi_A4 = tf.placeholder(tf.float32, [A4.shape[0], A4.shape[1]])
    Phi_A5 = tf.placeholder(tf.float32, [A5.shape[0], A5.shape[1]])
    Phi_A6 = tf.placeholder(tf.float32, [A6.shape[0], A6.shape[1]])
    Phi_A7 = tf.placeholder(tf.float32, [A7.shape[0], A7.shape[1]])
    Phi_A1_pinv = tf.placeholder(tf.float32, [A1_pinv.shape[0], A1_pinv.shape[1]])
    Phi_A2_pinv = tf.placeholder(tf.float32, [A2_pinv.shape[0], A2_pinv.shape[1]])
    Phi_A3_pinv = tf.placeholder(tf.float32, [A3_pinv.shape[0], A3_pinv.shape[1]])
    Phi_A4_pinv = tf.placeholder(tf.float32, [A4_pinv.shape[0], A4_pinv.shape[1]])
    Phi_A5_pinv = tf.placeholder(tf.float32, [A5_pinv.shape[0], A5_pinv.shape[1]])
    Phi_A6_pinv = tf.placeholder(tf.float32, [A6_pinv.shape[0], A6_pinv.shape[1]])
    Phi_A7_pinv = tf.placeholder(tf.float32, [A7_pinv.shape[0], A7_pinv.shape[1]])
    Initial_image = []
    Initial_PhiTY = []
    Initial_PhiTPhi = []
    num = 0
    Block_loss = 0
    for i in range(Block_num_m):  
        Initial_image_row = []
        Initial_PhiTY_row = []
        for j in range(Block_num_n):
            Block_output = []
            Block_output_PhiTY = []
            Block_output_PhiTPhi = []
            Block_image = x_image[:, i * Block_size:(i + 1) * Block_size, j * Block_size:(j + 1) * Block_size, :]

            [y_Block_pred_1, PhiTY_1, PhiTPhi_1] = Initial_Net(Block_image, rate_list[0], Phi_A1, Phi_A1_pinv)
            [y_Block_pred_2, PhiTY_2, PhiTPhi_2] = Initial_Net(Block_image, rate_list[1], Phi_A2, Phi_A2_pinv)
            [y_Block_pred_3, PhiTY_3, PhiTPhi_3] = Initial_Net(Block_image, rate_list[2], Phi_A3, Phi_A3_pinv)
            [y_Block_pred_4, PhiTY_4, PhiTPhi_4] = Initial_Net(Block_image, rate_list[3], Phi_A4, Phi_A4_pinv)
            [y_Block_pred_5, PhiTY_5, PhiTPhi_5] = Initial_Net(Block_image, rate_list[4], Phi_A5, Phi_A5_pinv)
            [y_Block_pred_6, PhiTY_6, PhiTPhi_6] = Initial_Net(Block_image, rate_list[5], Phi_A6, Phi_A6_pinv)
            [y_Block_pred_7, PhiTY_7, PhiTPhi_7] = Initial_Net(Block_image, rate_list[6], Phi_A7, Phi_A7_pinv)

            Block_output.append(y_Block_pred_1)
            Block_output.append(y_Block_pred_2)
            Block_output.append(y_Block_pred_3)
            Block_output.append(y_Block_pred_4)
            Block_output.append(y_Block_pred_5)
            Block_output.append(y_Block_pred_6)
            Block_output.append(y_Block_pred_7)

            Block_output_PhiTY.append(PhiTY_1)
            Block_output_PhiTY.append(PhiTY_2)
            Block_output_PhiTY.append(PhiTY_3)
            Block_output_PhiTY.append(PhiTY_4)
            Block_output_PhiTY.append(PhiTY_5)
            Block_output_PhiTY.append(PhiTY_6)
            Block_output_PhiTY.append(PhiTY_7)

            Block_output_PhiTPhi.append(PhiTPhi_1)
            Block_output_PhiTPhi.append(PhiTPhi_2)
            Block_output_PhiTPhi.append(PhiTPhi_3)
            Block_output_PhiTPhi.append(PhiTPhi_4)
            Block_output_PhiTPhi.append(PhiTPhi_5)
            Block_output_PhiTPhi.append(PhiTPhi_6)
            Block_output_PhiTPhi.append(PhiTPhi_7)

            Block_output = tf.convert_to_tensor(Block_output) 
            Block_output = tf.reshape(Block_output, [-1, Block_size * Block_size]) 
            Block_pred = tf.matmul(Onehot[:, num, :], Block_output)
            Block_pred = tf.reshape(Block_pred, [-1, Block_size, Block_size, 1])  

            Block_output_PhiTY = tf.squeeze(tf.convert_to_tensor(Block_output_PhiTY)) 
            Block_PhiTY = tf.matmul(Onehot[:, num, :], Block_output_PhiTY) 
            Block_PhiTY = tf.reshape(Block_PhiTY, [-1, Block_size, Block_size, 1]) 

            Block_output_PhiTPhi = tf.convert_to_tensor(Block_output_PhiTPhi) 
            Block_output_PhiTPhi = tf.reshape(Block_output_PhiTPhi, [Block_output_PhiTPhi.shape[0],
                                                                     Block_output_PhiTPhi.shape[1] *
                                                                     Block_output_PhiTPhi.shape[2]]) 
            Block_PhiTPhi = tf.matmul(Onehot[:, num, :], Block_output_PhiTPhi) 
            Initial_PhiTPhi.append(Block_PhiTPhi)

            Block_loss = Block_loss + tf.reduce_sum(tf.square(Block_image - Block_pred)) / 2 / batch_size
            num = num + 1
            if j == 0:
                Initial_image_row = Block_pred
                Initial_PhiTY_row = Block_PhiTY
            else:
                Initial_image_row = tf.concat([Initial_image_row, Block_pred], 2)
                Initial_PhiTY_row = tf.concat([Initial_PhiTY_row, Block_PhiTY], 2)
        if i == 0:
            Initial_image = Initial_image_row
            Initial_PhiTY = Initial_PhiTY_row
        else:
            Initial_image = tf.concat([Initial_image, Initial_image_row], 1)
            Initial_PhiTY = tf.concat([Initial_PhiTY, Initial_PhiTY_row], 1)

    y_pred_ini = Initial_image  
    y_pred, y_medium = Deep_Net(Initial_image, Initial_PhiTY, Initial_PhiTPhi, PhaseNumber)

    tvars = tf.trainable_variables()
    params = list(filter(lambda x: x.name.startswith('deep_conv_'), tvars))
    params_w2 = list(filter(lambda x: x.name.split('/')[-1].startswith('w2'), tvars))
    params_w1 = list(filter(lambda x: x.name.split('/')[-1].startswith('w1'), tvars))
    optimizer = tf.train.AdamOptimizer(1e-5)
    Train_loss = tf.reduce_sum(tf.square(x_image - y_pred)) / 2 / batch_size  
    grads_vars = optimizer.compute_gradients(Train_loss, var_list=params + params_w2)
    Train_op = optimizer.apply_gradients(grads_vars)

    print('把计算图写入事件文件')
    writer = tf.summary.FileWriter(logdir='Logs', graph=tf.get_default_graph())
    writer.close()

    # 启动会话
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=300)
    saver_ini = tf.train.Saver(var_list=params_w1)
    sess = tf.Session()
    sess.run(init)

    model_dir = 'D:\DeepBCS-master\Train\Train\Train_model_40layers'
    model_dir_ini = 'D:\DeepBCS-master\Train\Pre-train\Pretrain_model'

    cpkt_model_number_ini = 50
    saver_ini.restore(sess, '%s/CS_Saved_Model_%d.cpkt' % (model_dir_ini, cpkt_model_number_ini))

    cpkt_model_number = 50
    saver.restore(sess, '%s/CS_Saved_Model_%d.cpkt' % (model_dir, cpkt_model_number))

    if mode == 'test':
        PSNR_All = np.zeros([1, nrtest], dtype=np.float32)
        SSIM_All = np.zeros([1, nrtest], dtype=np.float32)
        for i in range(nrtest):
            test_x = np.float32(np.array(Image.open(filepaths[i])))
            test_Iorg = test_x / 255.0
            batch_x = test_Iorg[np.newaxis, :, :, np.newaxis]
            if nrtest == 1:
                batch_onehot = Block_onehot
            else:
                batch_onehot = Block_onehot[i, :, :]
                batch_onehot = batch_onehot[np.newaxis, :, :]
                        
            feed_dict = {x_image: batch_x, Onehot: batch_onehot, Phi_A1: A1, Phi_A2: A2, Phi_A3: A3, Phi_A4: A4,
                         Phi_A5: A5, Phi_A6: A6, Phi_A7: A7, Phi_A1_pinv: A1_pinv, Phi_A2_pinv: A2_pinv,
                         Phi_A3_pinv: A3_pinv, Phi_A4_pinv: A4_pinv, Phi_A5_pinv: A5_pinv, Phi_A6_pinv: A6_pinv,
                         Phi_A7_pinv: A7_pinv}
                         
            Prediction = sess.run(y_pred, feed_dict=feed_dict)

            rec_PSNR = psnr(Prediction * 255, batch_x * 255)

            Prediction = np.squeeze(Prediction)
            batch_x = np.squeeze(batch_x)
            rec_SSIM = compute_ssim(Prediction * 255, batch_x * 255)
            print("PSNR is %.2f, SSIM is %.4f" %(rec_PSNR, rec_SSIM))

            PSNR_All[0, i] = rec_PSNR
            SSIM_All[0, i] = rec_SSIM

            x_im_rec = Image.fromarray(np.clip(Prediction * 255, 0, 255).astype(np.uint8))
            img_rec_name = "%s" % (filepaths[i])
            save_dir = save_paths + '%s_%f.png' % (os.path.splitext(img_rec_name.split('\\')[-1])[0], rec_PSNR)
            x_im_rec.save(save_dir)

