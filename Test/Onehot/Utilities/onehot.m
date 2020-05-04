
function Block_onehot = onehot(Block_sampling_rate, C)

[m,n,s]=size(Block_sampling_rate);
Block_onehot=zeros(m*n,C,s);

for k=1:s
    num=1;
    for i=1:m
        for j=1:n  
            if Block_sampling_rate(i,j,k)==0.01
                Block_onehot(num,:,k)=[1,0,0,0,0,0,0];
            elseif Block_sampling_rate(i,j,k)==0.03
                Block_onehot(num,:,k)=[0,1,0,0,0,0,0];
            elseif Block_sampling_rate(i,j,k)==0.05
                Block_onehot(num,:,k)=[0,0,1,0,0,0,0];
            elseif Block_sampling_rate(i,j,k)==0.1
                Block_onehot(num,:,k)=[0,0,0,1,0,0,0];
            elseif Block_sampling_rate(i,j,k)==0.2
                Block_onehot(num,:,k)=[0,0,0,0,1,0,0];
            elseif Block_sampling_rate(i,j,k)==0.3
                Block_onehot(num,:,k)=[0,0,0,0,0,1,0];
            elseif Block_sampling_rate(i,j,k)==0.4
                Block_onehot(num,:,k)=[0,0,0,0,0,0,1];
            end
            num=num+1;                  
        end
    end
end