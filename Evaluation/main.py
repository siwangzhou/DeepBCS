import tensorflow as tf
import numpy as np
import os
from PIL import Image
import train_cifar10_vgg16


sample_rate =0.1
image_size = 512
image_size1 =512
mode='test'
file_name = 'baby'
traindata_path='D:/DeepBCS-master/DataSets/TestData/Set5'
testdata_path='D:/DeepBCS-master/DataSets/Prediction/%s_FSR' %sample_rate
#name=['baby','bird','butterfly','head','woman']

def load_data(dataset_trainpath,dataset_testpath):   
    
    train_x=[]
    test_x=[]
    files = os.listdir(dataset_trainpath)  
    test_files = os.listdir(dataset_testpath)

    for f in files:
        if f.startswith(file_name) : 
            print(f)
            fname = dataset_trainpath + '/' + f
            img = Image.open(fname)
            img = np.array(img.convert('L'))  
            img = np.float32(img)
           
            train_x=img/255.0      
    
    for f1 in test_files:
        if f1.startswith(file_name) :
            print(f1)
            fname1 = dataset_testpath + '/' + f1
            img1 = Image.open(fname1)
            img1 = np.array(img1.convert('L'))  
            img1 = np.float32(img1)
            test_x.append((img1/255.0))
   
    return train_x,test_x

def run_testing(sess, train_x, test_x):
    
    train_x = train_x.eval()
    test_x = test_x.eval()
#             
    p_loss=sess.run([loss_],feed_dict={ x_image:train_x,nx_image:test_x,keep_prob:1.0,train_flag:False})
   
    return p_loss


with tf.Graph().as_default():

    x_image = tf.placeholder(tf.float32, [None, image_size,image_size1,1], name="x_image")
    nx_image = tf.placeholder(tf.float32, [None, image_size,image_size1,1], name="nx_image")
    train_flag = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32,name = "keep_prob")
    
    y_pred = train_cifar10_vgg16.Vgg_16(x_image, keep_prob,train_flag,False)  
    y_pred1 = train_cifar10_vgg16.Vgg_16(nx_image, keep_prob,train_flag,True)
   
    train_loss_1=tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_pred1)))    
    train_loss_2=tf.sqrt(tf.reduce_sum(tf.square(y_pred ))) 
       
    loss_ =train_loss_1 / train_loss_2

    
    tvars = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    params_w2 =bn_moving_vars+list(filter(lambda x: x.name.startswith('VGG16/'), tvars)) 
  
    saver_AdvCNN = tf.train.Saver(var_list=params_w2)      
    init=tf.global_variables_initializer() 
    config_proto = tf.ConfigProto()
    config_proto.gpu_options.allow_growth=True

    sess = tf.InteractiveSession(config=config_proto)
    sess.run(init)    

    #恢复部分参数 
    model_path='E:/pd/CSCD/VGG16_model/cifar10_1/' 
    saver_AdvCNN.restore(sess, model_path +'AdvCNN_Saved_Model_298.cpkt')
          
    if mode == 'test': 
        train_x, test_x =load_data(traindata_path,testdata_path)
        for i in range(1):
            train_x =tf.reshape(train_x,[-1,image_size,image_size1,1])
            test_x1 =tf.reshape(test_x[i],[-1,image_size,image_size1,1])

            perceptual_loss, = run_testing(sess, train_x, test_x1)            
            print("perceptual_loss: %.4f" % (perceptual_loss)) 
        