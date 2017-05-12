clear all
close all
clc

addpath(genpath([pwd,'/toolboxes']));
addpath(genpath([pwd,'/toolbox_graph']));
addpath(genpath([pwd,'/geom3d']));
addpath(genpath([pwd,'/subdivide_tri']));
addpath(genpath([pwd,'/gptoolbox']));

% ,
folders = {'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'car', ...
        'chair', 'cup', 'curtain', 'desk', 'door', 'dresser', ...
        'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', ...
        'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', ...
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', ...
        'wardrobe', 'xbox'}
    
folders = {'bottle', 'bowl', 'cone', 'flower_pot', 'vase'}

para.flag = 0
para.scale = 1
para.n = 10000 % 40
para.Nfill = 10
para.sphere_radius = 2.0;


parfor thresdi = 1:length(folders)
        
    folder = ['~/dataset/ModelNet40/', folders{thresdi}, '/train'];
    dest_folder = ['~/dataset/processed_data3/', folders{thresdi}, '/train'];
    pcl_folder = ['~/dataset/pcl/', folders{thresdi}, '/train'];
    
    if ~exist(dest_folder)
        mkdir(dest_folder);
    end
    
    process_folder(folder, pcl_folder, dest_folder, para);
    
    folder = ['~/dataset/ModelNet40/', folders{thresdi}, '/test'];
    dest_folder = ['~/dataset/processed_data3/', folders{thresdi}, '/test'];
    pcl_folder = ['~/dataset/pcl/', folders{thresdi}, '/test'];
    
    if ~exist(dest_folder)
        mkdir(dest_folder);
    end
    
    process_folder(folder, pcl_folder, dest_folder, para);
    
end

