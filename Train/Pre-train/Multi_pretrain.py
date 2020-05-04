# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import glob
from PIL import Image

batch_size=1
Block_size=32
image_channel=1
EpochNum=2000
rate_list=[0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4]
k=len(rate_list)

train_path = 'D:/DeepBCS-master/DataSets\TrainData'
train_file = glob.glob(train_path + '/*.jpg')
nrtrain=len(train_file)
image_size_row=96
image_size_col=96
Block_num_m=np.int(image_size_row/Block_size)
Block_num_n=np.int(image_size_col/Block_size)

train_onehot_path = 'D:/DeepBCS-master/Train/Onehot/train_0.1_7'
Onehot_name = sio.loadmat(train_onehot_path)
Block_onehot =np.transpose(np.float32(Onehot_name['Block_onehot']),(2,0,1))
    
def generatebatch_image(file, number):
    batch_img=[]
    for i in range(number):
        img = np.float32(np.array(Image.open(file[i])))
        batch_img.append(img)
    batch_img=np.array(batch_img)
    return batch_img[:,:,:,np.newaxis]
    
def WeightsVariable(shape,name='w',trainable=True):
    if len(shape)==4:
        N = shape[0]*shape[1]*(shape[2]+shape[3])/2
    else:
        N = shape[0]/2
    initial = tf.random_normal(shape=shape,stddev=np.sqrt(2.0/N)) 
    return tf.get_variable(name,initializer=initial,trainable=trainable)
        
def Conv2d_no_bias(x, w,  stride=1, padding='SAME'):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)
    
    
def Initial_Net(Block_image, Block_sampling_rate):  
    nB = np.int32(np.round(Block_size * Block_size * Block_sampling_rate))
    with tf.variable_scope('sample_initial_'+str(Block_sampling_rate)+str(nB), reuse=tf.AUTO_REUSE):        
        weight1=WeightsVariable(shape=[Block_size, Block_size, image_channel, nB],name='w1')
        conv1=Conv2d_no_bias(Block_image, weight1, stride=Block_size, padding='VALID')
        
        weight2=WeightsVariable(shape=[1, 1, nB, Block_size*Block_size],name='w2')
        conv2=Conv2d_no_bias(conv1, weight2, stride=1,padding='VALID')
        y_Block_pred=tf.reshape(conv2, [-1, Block_size, Block_size, image_channel])
    return  weight1, y_Block_pred
    

with tf.Graph().as_default():
    x_image=tf.placeholder(tf.float32, [None, image_size_row, image_size_col, image_channel])
    Onehot=tf.placeholder(tf.float32, [None, Block_onehot.shape[1], k])
    Initial_image=[]
    num=0
    for i in range(Block_num_m):  
        Initial_image_row=[]        
        for j in range(Block_num_n):           
            Block_output=[]
            Block_weights_output=[]
            Block_image=x_image[:,i*Block_size:(i+1)*Block_size, j*Block_size:(j+1)*Block_size, :]
            for Block_sampling_rate in rate_list:
                [Block_weights, Block_intial_image]=Initial_Net(Block_image, Block_sampling_rate)
                Block_output.append(Block_intial_image)  
                Block_weights_output.append(Block_weights)
              
            Block_output=tf.convert_to_tensor(Block_output) 
            Block_output=tf.transpose(tf.reshape(Block_output,[k, -1, Block_size*Block_size]),[1,0,2]) 
            Block_pred=tf.matmul(Onehot[:,num,tf.newaxis,:], Block_output)
            Block_pred=tf.reshape(Block_pred,[-1, Block_size, Block_size])
            num=num+1
            if j==0:
                Initial_image_row=Block_pred   
            else:
                Initial_image_row=tf.concat([Initial_image_row, Block_pred], 2)                
        if i==0: 
            Initial_image=Initial_image_row
        else:
            Initial_image=tf.concat([Initial_image, Initial_image_row], 1) 
    
    y_pred=tf.expand_dims(Initial_image, -1)

    optimizer = tf.train.AdamOptimizer(1e-5)
    train_loss = tf.reduce_sum(tf.square(x_image - y_pred))/2/batch_size 

    grads_vars = optimizer.compute_gradients( train_loss)
    Train_op = optimizer.apply_gradients( grads_vars)
    
    print('把计算图写入事件文件')
    writer=tf.summary.FileWriter(logdir='Logs', graph=tf.get_default_graph())
    writer.close()
    
    #启动会话 
    init=tf.global_variables_initializer()    
    saver=tf.train.Saver(max_to_keep=300)
    sess=tf.Session()
    sess.run(init) 
    
    print("Strart Training..")
    
    model_dir = './Pretrain_model' 
    cpkt_model_number = 50
    saver.restore(sess, '%s/CS_Saved_Model_%d.cpkt' % (model_dir, cpkt_model_number))

    output_file_name = "%s/Log_output.txt" % (model_dir)
    
    train_x=generatebatch_image(train_file, nrtrain)
    Iorg = train_x/255.0       
    for epoch_i in range(1, EpochNum+1):
        Loss=0.
        randidx_all = np.random.permutation(nrtrain)   
        for batch_i in range(nrtrain // batch_size):    
            randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]
            batch_x = Iorg[randidx, :, :, :]
            batch_onehot = Block_onehot[randidx,:,:]
            feed_dict = {x_image: batch_x, Onehot: batch_onehot}
            sess.run(Train_op, feed_dict=feed_dict)
            loss=sess.run(train_loss, feed_dict=feed_dict)
            Loss=Loss+loss
        output_data = "[%02d/%02d] cost: %.4f\n" % (epoch_i, EpochNum, Loss/nrtrain)
        print(output_data)

        output_file = open(output_file_name, 'a')
        output_file.write(output_data)
        output_file.close()
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        saver.save(sess, '%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
         
    print("Training Finished")
    
    #保存测量矩阵
    weights = sess.run(Block_weights_output, feed_dict=feed_dict)
    # B=np.squeeze(weights[0])   
    # C=np.transpose(np.reshape(B,[1024,51]))  
    # sio.savemat(save_paths+'A1.mat',mdict={'A1':C}) 
