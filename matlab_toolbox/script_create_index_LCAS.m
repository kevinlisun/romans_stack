clear all
close all
clc

dataset_dir = '/home/kevin/dataset/LCAS/val'

phase = 'pcd'

fid = fopen([dataset_dir, '/', phase, '.txt'],'w');

folders = dir(dataset_dir);

folder_dir = [dataset_dir, '/', phase];

files = dir(folder_dir);
last = [];

for j = 1:length(files)
    
    if files(j).isdir
        continue;
    end
    
    
    filej = strsplit(files(j).name, '.');
    file_name = [filej{1}, '.',filej{2}];
    
    fprintf(fid,'%s\n', file_name);
end



