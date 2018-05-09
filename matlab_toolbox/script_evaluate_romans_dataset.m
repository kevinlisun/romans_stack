clear all
close all
clc

dataset_dir = '/home/kevin/romans_bagfiles/evaluation'
dest_dir = '/home/kevin/romans_bagfiles/evaluation/visualization'

if ~exist(dest_dir)
    mkdir(dest_dir)
end
    

sets = {'test1', 'test2', 'test4', 'test5'}

shift_x = -10
shift_y = -1

% % F_SCORE = zeros(41,41);

% % for shift_x = -20:20
% %     for shift_y = -20:20
        
        disp(['precessing ', num2str(shift_x), ' ', num2str(shift_y), '...'])

Prediction = [];
GroundTruth = [];

for i = 1:length(sets)
    folder = [ dataset_dir, '/', sets{i} ];
    
    files = dir([folder, '/rgb']);
    
    for j = 3:length(files)
        file_name = files(j).name;
        file_name = strsplit(file_name, '.');
        file_name = file_name{1};
        
        disp([sets{i}, '_', file_name])
        
        rgb_file = [folder, '/rgb/', file_name, '.jpg'];
        depth_file = [folder, '/depth/', file_name, '.png'];
        prediction_file = [folder, '/prediction/', file_name, '.mat'];
        ground_truth_file = [folder, '/ground_truth/', file_name, '.png'];
        
        rgb = imread(rgb_file);
        depth = imread(depth_file);
        
        X = imread(ground_truth_file);
        ground_truth = X(:, :, 1);
        ground_truth = bitor(ground_truth, bitshift(X(:, :, 2), 8));
        ground_truth = bitor(ground_truth, bitshift(X(:, :, 3), 16));
        [ground_truth] = shift(ground_truth, shift_x, shift_y);
        
        load(prediction_file);
        prediction = semantic_map;
        
        depth = double(depth);
        ground_truth = double(ground_truth);
        prediction = double(prediction);
        
        ground_truth(depth==0) = NaN;
        prediction(depth==0) = NaN;
        
        Prediction = [ Prediction; prediction(:) ];
        GroundTruth = [ GroundTruth; ground_truth(:) ];
        
        figure(1)
        subplot(2,2,1)
        imagesc(rgb)
        subplot(2,2,3)
        imagesc(ground_truth)
        subplot(2,2,4)
        imagesc(prediction)
        subplot(2,2,2)
        imagesc(prediction==ground_truth)
        
        %pause;
        
        gt = ground_truth;
        gt(isnan(gt)) = 0;
        pd = prediction;
        pd(isnan(pd)) = 0;
        save([dest_dir,'/', sets{i}, '_', file_name, '_gt.mat'], 'gt');
        save([dest_dir,'/', sets{i}, '_', file_name, '_pd.mat'], 'pd');
        save([dest_dir,'/', sets{i}, '_', file_name, '_depth.mat'], 'depth');
    end
    
end


C = 11;
[precision recall fscore] = evaluate_seg_result(Prediction, GroundTruth, C)
precision*100
recall*100
fscore*100
% % F_SCORE(shift_y+21,shift_x+21) = fscore(end);
% %     end
% % end