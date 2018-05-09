clear
clc

split_rate = 0.05

dataset_dir = '/home/kevin/dataset/washington_rgbd_dataset'
dataset_dir2 = '/home/kevin/dataset/ws_exp'

cmd = ['rm -rf ', dataset_dir2];
system(cmd);

cmd = ['mkdir ', dataset_dir2];
system(cmd);
cmd = ['mkdir ', dataset_dir2, '/labelled', ' & ', 'mkdir ', dataset_dir2, '/unlabelled', ' & ', 'mkdir ', dataset_dir2, '/gp_labelled'];
system(cmd);

cat = {'apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl', 'calculator', 'camera', 'cap', 'cell_phone', 'cereal_box', 'coffee_mug', 'comb', 'dry_battery', 'flashlight', 'food_bag', 'food_box', 'food_can', 'food_cup', 'food_jar', 'garlic', 'glue_stick', 'greens', 'hand_towel', 'instant_noodles', 'keyboard', 'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook', 'onion', 'orange', 'peach', 'pear', 'pitcher', 'plate', 'pliers', 'potato', 'rubber_eraser', 'scissors', 'shampoo', 'soda_can', 'sponge', 'stapler', 'tomato', 'toothbrush', 'toothpaste', 'water_bottle'}


for i = 1:length(cat)
    cmd = [ 'mkdir ', dataset_dir2, '/labelled/', cat{i} ]
    system(cmd)
end

fid = fopen([dataset_dir, '/', 'train', '.txt'],'r');

tline = fgetl(fid);

files = []

while ischar(tline)
    files = [files; cellstr(tline)];
    tline = fgetl(fid);
end

file_num = length(files);

index = randperm(file_num);

labelled_num = 1750; %round(file_num*split_rate);

for i = 1:length(index)
    rgb_file = [ files{index(i)}, '_crop.png' ]
    depth_file = [ files{index(i)}, '_depth_HHA.png' ]
    
    if i <= labelled_num
        folder = [ dataset_dir2, '/', 'labelled' ];
        
        rgb_file2 = strsplit(files{index(i)}, '/');
        depth_file2 = strsplit(files{index(i)}, '/');
        
        cmd = [ 'cp ', dataset_dir, '/', rgb_file, ' ', folder, '/', rgb_file2{1}, '/', rgb_file2{3}, '.png' ];
        system(cmd);
        
        cmd = [ 'cp ', dataset_dir, '/', depth_file, ' ', folder, '/', depth_file2{1}, '/', depth_file2{3}, '_depth.png' ];
        system(cmd);  
    end
    
    folder = [ dataset_dir2, '/', 'unlabelled' ];
    
    rgb_file2 = strsplit(files{index(i)}, '/');
    depth_file2 = strsplit(files{index(i)}, '/');
    
    cmd = [ 'cp ', dataset_dir, '/', rgb_file, ' ', folder, '/', rgb_file2{3}, '.png' ];
    system(cmd);
    
    cmd = [ 'cp ', dataset_dir, '/', depth_file, ' ', folder, '/', depth_file2{3}, '_depth.png' ];
    system(cmd); 
    
    
end
    
    
        



