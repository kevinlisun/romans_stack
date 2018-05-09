clear all
close all
clc

dataset_dir = '/home/kevin/dataset/washington_rgbd_dataset'

folders = dir(dataset_dir);

for i = 3:length(folders)
    folder = folders(i).name;
    
    subfolders = dir([dataset_dir,'/', folder]);
    
    for j = 3:length(subfolders)
        subfolder = subfolders(j).name;
        
        depth_files = dir(fullfile([dataset_dir,'/', folder, '/', subfolder, '/*_depth.png']));
        
        for k = 1:length(depth_files)
            depth_file = depth_files(k).name;
            
            depth_map = imread([dataset_dir,'/', folder, '/', subfolder, '/', depth_file]);
            
            loc_file = strsplit(depth_file, '.');
            loc_file{1} = loc_file{1}(1:end-5);
            loc_file = [loc_file{1}, 'loc.txt'];
            
            [loc_x, loc_y] = textread([dataset_dir,'/', folder, '/', subfolder, '/', loc_file], '%d,%d');
        
            HHA = depth2HHA(depth_map, [loc_x, loc_y]);
            
            tmp = strsplit(depth_file, '.');
            HHA_file = [tmp{1}, '_HHA.png'];

            imwrite(HHA, [dataset_dir,'/', folder, '/', subfolder, '/', HHA_file])
            disp(['wrote', dataset_dir,'/', folder, '/', subfolder, '/', HHA_file]);
            
        end
    end
    
end