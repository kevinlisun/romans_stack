clear all
close all
clc

dataset_dir = '~/dataset/processed_data'
level = 'object' % or frame
phase = 'train'

fid = fopen([dataset_dir, '/', phase, '_', level, '.txt'],'w');

folders = dir(dataset_dir);

for i = 3:length(folders)
    
    if ~folders(i).isdir
        continue;
    end
    
    disp(['processing ', folders(i).name, '...'])
    
    label = findLabel(folders(i).name);
    
    sub_dir = [dataset_dir, '/', folders(i).name, '/', phase];
    
    files = dir(sub_dir);
    last = [];
    
    for j = 1:length(files)
        
        if files(j).isdir
            continue;
        end
        
        if strcmp(level, 'object')
            filej = strsplit(files(j).name, '_');
            filej(end) = [];
            
            for k = 1:length(filej)
                if k == 1
                    file_j = filej{k};
                else
                    file_j = [ file_j, '_', filej{k} ] ;
                end
            end
            file_name = [folders(i).name, '/', phase, '/', file_j];
            
            if ~strcmp(file_name, last)
                fprintf(fid,'%s %i\n', file_name, label);
                last = file_name;
            end
            
        else
            
            filej = strsplit(files(j).name, '.');
            filej = filej{1};
            file_name = [folders(i).name, '/', phase, '/', filej];
            
            fprintf(fid,'%s %i\n', file_name, label);
        end
    end

end

    