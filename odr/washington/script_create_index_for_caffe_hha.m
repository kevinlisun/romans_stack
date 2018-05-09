clear all
close all
clc

dataset_dir = '~/dataset/ws_exp/gp_labelled'
phase = 'train'

fid = fopen([dataset_dir, '/', phase, '.txt'],'w');

folders = dir(dataset_dir);

for i = 3:length(folders)
    
    if ~folders(i).isdir
        continue;
    end
    
    disp(['processing ', folders(i).name, '...'])
    
    label = findLabel(folders(i).name);
    
    sub_dir = [dataset_dir, '/', folders(i).name];
    
    files = dir(sub_dir);
    files = dir(fullfile([sub_dir, '/*_depth.png']));
    
    last = [];
    
    for j = 1:length(files)
        
        if files(j).isdir
            continue;
        end
        
        filej = strsplit(files(j).name, '.');
        
        if strcmp(filej{2}, 'png')
            
            filej = filej{1};
            file_name = [folders(i).name, '/', filej(1:end-6)];
            
            fprintf(fid,'%s %i\n', file_name, label);
        end
    end
end


