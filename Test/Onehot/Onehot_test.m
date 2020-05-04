clear all
clc
addpath('D:\DeepBCS-master\Test\Onehot\Utilities');

Sampling_rate=0.3; 
C=7;  

%prepare onehot
AdaRate=num2str(Sampling_rate); 
RateDir=['.\Set5\', AdaRate, '_7_3\', 'sampling_rate_woman_', AdaRate, '.mat'];
load (RateDir, 'Block_sampling_rate')
Block_onehot = onehot(Block_sampling_rate, C);