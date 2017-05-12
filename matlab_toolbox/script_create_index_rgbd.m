clear all
close all
clc

dataset_dir = '~/dataset/rgbd'
phase = 'train'

fid = fopen([dataset_dir, '/', phase, '.txt'],'w');

folders = dir(dataset_dir);

for i = 3:length(folders)
    
    if ~folders(i).isdir
        continue;
    end
    
    disp(['processing ', folders(i).name, '...'])
    
    label = findLabel2(folders(i).name);
    
    sub_dir = [dataset_dir, '/', folders(i).name, '/', phase];
    
    files = dir(sub_dir);
    last = [];
    
    for j = 1:length(files)
        
        if files(j).isdir
            continue;
        end
        

        filej = strsplit(files(j).name, '.');
        extension = filej{end};
        filej = filej{1};
        file_name = [folders(i).name, '/', phase, '/', filej];
        
        if strcmp(extension, 'jpg')
            fprintf(fid,'%s %i\n', file_name, label);
        end
        
    end

end

    