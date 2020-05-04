function smap=saliency(X)
[a,b]=size(X);
%% 
P=fft2(X); 
myLogAmplitude = log(abs(P));
myPhase = angle(P);
mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate'); 
smap = abs(ifft2(exp(mySpectralResidual + i*myPhase))).^2;
smap = mat2gray(imfilter(smap, fspecial('gaussian', [10, 10], 3)));

