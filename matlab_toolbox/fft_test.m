I = depth_map;

% normalize the image and conver to doubleI
I = double(mat2gray(I));

% Resize the image
I = imresize(I, [256 256]);

% get the size of the image
[rows,cols] = size(I);

% apply FFT
f = fftshift(fft2(I));

% used to plot the image
fLog = log(1 + abs(f));

% filter by a range based on fLog

filter = (fLog < .9*max(fLog(:)) ) & (fLog > .2*max(fLog(:)) );

B = abs(ifft2(f.*filter));

colormap(gray)
subplot(2,2,1),surf(I); view(2); camlight right; lighting phong; shading interp;; title('Cleaned Image'); title('Original Image')
subplot(2,2,2),imagesc(fLog); title('Fourier Image')
subplot(2,2,3),imagesc(filter); title('Zeroed Fourier Image')
subplot(2,2,4),surf(B); view(2); camlight right; lighting phong; shading interp; title('Cleaned Image')
annotation('textbox', [0 0.9 1 0.1], ...
    'String', 'Fourier Analysis on Clown Image', ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center', ...
    'FontSize', 15, ...
    'FontWeight', 'bold')

pause
close all 

I = depth_map;
J = dct2(I);
%subplot(1,3,1); imshow(log(abs(J)),[]), colormap(gca,jet(64)), colorbar
N = J; N(abs(N) > 0.1) = 0;
noise = N;
J(abs(J) < 0.1) = 0;
S = idct2(J);
K = idct2(N);

subplot(1,3,1); surf(I); view(2); camlight right; lighting phong; shading interp;
subplot(1,3,2); surf(S); view(2); camlight right; lighting phong; shading interp;
subplot(1,3,3); surf(K); view(2); camlight right; lighting phong; shading interp;