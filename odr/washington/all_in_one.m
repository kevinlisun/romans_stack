

cat = {'apple', 'ball', 'banana', 'bell_pepper', 'binder', 'bowl', 'calculator', 'camera', 'cap', 'cell_phone', 'cereal_box', 'coffee_mug', 'comb', 'dry_battery', 'flashlight', 'food_bag', 'food_box', 'food_can', 'food_cup', 'food_jar', 'garlic', 'glue_stick', 'greens', 'hand_towel', 'instant_noodles', 'keyboard', 'kleenex', 'lemon', 'lightbulb', 'lime', 'marker', 'mushroom', 'notebook', 'onion', 'orange', 'peach', 'pear', 'pitcher', 'plate', 'pliers', 'potato', 'rubber_eraser', 'scissors', 'shampoo', 'soda_can', 'sponge', 'stapler', 'tomato', 'toothbrush', 'toothpaste', 'water_bottle'}

cmd = 'cd /home/kevin/catkin_ws/src/romans_stack/odr/scripts'
system(cmd)

cmd = []

fid = fopen('cat_list2.txt','w');
    
for i = length(cat):-1:1
    
%     cmd1 = [ 'rosrun odr train_classifier_ws.py gpu rgbd ', cat{i} ]
%     
%     cmd2 = [ 'rosrun odr classify_meta_data_ws.py gpu rgbd ', cat{i} ]
    
    fprintf(fid,'%s\n', cat{i});
    
%     fprintf(fid,'%s\n', cmd2);


end