Experimental environments: 
TensorFlow 1.4 + Python3.6; Matlab R2014a; Windows

### Datasets
The training dataset, consisting of 89600 images of size 96*96, is stored in the subdirectory ¡°TrainData¡±. 

The test datasets are stored in the subdirectory ¡°TestData¡±, which includes three widely used benchmark datasets, Set5, Set11 and BSD100. Notice that for BSD100 only the central 320*320 part of each image is extracted as test image.

### Train
The onehot label file for training dataset is ¡°Onehot\train_0.1_7.mat¡±. For training, we prepare onehot label for each block of the image in the training dataset, so that the training images can go through the multi-channel network. A onehot label corresponds to k channels in the multi-channel network. In the experiment we set k = 7, and note that k can be set to any other values. Each of the k channels is related to a sampling interval [T_c, T_c+1]. If the sampling rate assigned to a block falls into an interval, the third interval for example, then for this block the corresponding channel is labelled 1 and other channels are 0, i.e., the onehot label of this block is 0010000. The formula for assigning different sampling rates to the blocks is referred to the reference [12] in the manuscript. 

The Python file to pre-train the sampling matrix is ¡°Pre-train\Multi_pretrain.py¡±. The well-trained matrix and its pseudo-inverse are stored in ¡°Pre-train \phi_0.1_7.mat¡± and ¡°\Pre-train\phi_pinv_0.1_7.mat¡±, respectively. The well-trained network model is stored in the subdirectory ¡°Pre-train\Pretrain_model¡±. 

The Python source file to train the BCSnet and BCSnet-FSR models is ¡°Train\Multi_40layers.py¡±. The Python source file to train the BCSnet-WBA and BCSnet-WBA-FSR models is ¡°Train\Multi_40layer_WBA.py¡±. The corresponding trained models are stored in the subdirectories ¡°Train\Train_model_40layers¡± and ¡°Train\Train_model_40layers_WBA¡±, respectively.

### Test
The onehot label files for three test datasets are stored in the subdirectory ¡°Onehot¡±. Similar to the training process, we first prepare onehot labels for the test datasets, so that the testing images can enter the proposed multi-channel network. As of BCSNet-FSR and BCSNet-WBA-FSR approaches with FSR strategy, each block of the images has the same sampling rate. At this point, all blocks have the same onehot label. As of BCSNet and BCSNet-WBA, the different sampling rates of the blocks are pre-computed from the test images by mimicking the initial sampling techniques. The onehot labels for the blocks of each test image are then calculated similarly to the training process. 

The Python source files to run testing for BCSnet, BCSnet-WBA, BCSnet-FSR, and BCSnet-WBA-FSR approaches are ¡°Test_Adap\BCSnet.py¡±, ¡°Test_Adap\BCSnet-WBA.py¡±, ¡°Test_FSR\BCSnet-FSR.py¡±, ¡°Test_FSR\BCSnet-WBA-FSR.py¡±, respectively.

### Evaluation
Include two Python source files for evaluating the perceptual similarity index based on the pre-trained VGG-16 model.

Any questions, please contact us at swzhou@hnu.edu.cn.
