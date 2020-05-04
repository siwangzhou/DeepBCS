clear all
clc
addpath('D:\DeepBCS-master\Test\Onehot\Utilities');

Image_num=89600;
block_size=32;
Sampling_rate=0.1;
C=7;
num=zeros(1,C);
Block_sampling_rate=zeros(3,3,Image_num);

%sub-rate
for k=1:Image_num
    Im=imread(['D:\DeepBCS-master\DataSets\TrainData\',num2str(k),'.jpg']);
    [row,col]=size(Im); 
    Im_sal=saliency(Im);  

    Block_sal=zeros(row/block_size,col/block_size); 
    Im_sal_avg=mean(mean(Im_sal));

    for i=1:row/block_size
        for j=1:col/block_size
            Block_sal(i,j)=mean(mean(Im_sal((i-1)*block_size+1:i*block_size,(j-1)*block_size+1:j*block_size)));
        end
    end

    Block=zeros(row/block_size,col/block_size);
    for i=1:row/block_size
        for j=1:col/block_size
            Block(i,j)=(Sampling_rate*Block_sal(i,j))/Im_sal_avg;         
            %0.1-7 
            if Block(i,j)>0 && Block(i,j)<=0.03
                num(1,1)= num(1,1)+1;
                Block_sampling_rate(i,j,k)=0.01;
            elseif Block(i,j)>0.03 && Block(i,j)<=0.055
                num(1,2)= num(1,2)+1;
                Block_sampling_rate(i,j,k)=0.03;
            elseif Block(i,j)>0.055 && Block(i,j)<=0.08
                num(1,3)= num(1,3)+1;
                Block_sampling_rate(i,j,k)=0.05;
            elseif Block(i,j)>0.08 && Block(i,j)<=0.1
                num(1,4)= num(1,4)+1;
                Block_sampling_rate(i,j,k)=0.1;
            elseif Block(i,j)>0.1  && Block(i,j)<=0.125
                num(1,5)= num(1,5)+1;
                Block_sampling_rate(i,j,k)=0.2;
            elseif Block(i,j)>0.125  && Block(i,j)<=0.17
                num(1,6)= num(1,6)+1;
                Block_sampling_rate(i,j,k)=0.3;
            elseif Block(i,j)>0.17 
                num(1,7)= num(1,7)+1;
                Block_sampling_rate(i,j,k)=0.4;
            end 
        end
    end
end
Block_onehot = onehot(Block_sampling_rate, C);