clear
close all
clc

addpath('toolbox_nyu_depth_v2');

dataset_dir = '/home/kevin/dataset/washington_rgbd_dataset';
index_file = [ dataset_dir, '/test.txt' ];

fid = fopen(index_file);

tline = fgets(fid);
while ischar(tline)
    disp(tline)
    
    str = strsplit(tline, ' ');
    
    file_name = str{1}
    
    
    imgRgb = imread([dataset_dir, '/', file_name, '_crop.png']);
    imgDepthAbs = imread([dataset_dir, '/', file_name, '_depthcrop.png']);
    
    % Crop the images to include the areas where we have depth information.
    imgDepthFilled = fill_depth_cross_bf(imgRgb, double(imgDepthAbs));
    
% %     figure(1);
% %     subplot(1,3,1); imagesc(imgRgb);
% %     subplot(1,3,2); imagesc(imgDepthAbs);
% %     subplot(1,3,3); imagesc(imgDepthFilled);
    
    imwrite(uint16(imgDepthFilled), [dataset_dir, '/', file_name, '_depth.png']);
    
% %     pause(0.1)
    
    
    tline = fgets(fid);
end

fclose(fid);