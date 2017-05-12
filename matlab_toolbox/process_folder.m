function [] = process_folder(folder, pcl_folder, dest_folder, para)

n = para.n;
Nfill = para.Nfill;
flag = para.flag;
scale = para.scale;
sphere_radius = para.sphere_radius;

if ~exist(dest_folder)
    mkdir(dest_folder);
end

if ~exist(pcl_folder)
    mkdir(pcl_folder);
end

list = dir(folder)

for filei = 3:length(list)
    
    file = list(filei).name;
    
    object_name = strsplit(file, '.');
    object_name = object_name{1};
    
    %% load the mesh
    file_name = [folder, '/', file];
    
    [vertex, faces, UV, C, N] = readOFF( file_name );
    [points,I,B] = random_points_on_mesh(vertex, faces, n, 'Color', 'white', 'MaxIter', 10);
    
    if flag
        figure(1)
        clf;
        plot_mesh(vertex, faces);
        shading interp;
    end
    
    vertex = vertex';
    faces = faces';
    
    %% normalize point cloud
    % normalize object and NOT keep object ratio
    min_v = min(points, [], 1);
    max_v = max(points, [], 1);
    
    mean_v = mean(points, 1);
    std_v = std(points, 1);
    
    if sum((max_v - min_v)<=0.03) > 0
        disp('ERROR bad mesh! Abandon it!');
        continue;
    end

% %     % keep object ratio 
% %     [a b] = max(max_v - min_v);
% %     min_v = ones(1,3)*min_v(b);
% %     max_v = ones(1,3)*max_v(b);
    
    points = (points - repmat(mean_v, [size(points,1) 1])) ./ ( repmat(std_v, [size(points,1) 1]) );
    points = points / 3;
    
    min_p = min(points, [], 1);
    max_p = max(points, [], 1);
% %     mean_p = (max_p + min_p) / 2;
% %     points = points - repmat(mean_p,[size(points,1) 1]);
    
    % save point cloud
    pcl_file_name = [pcl_folder, '/', object_name, '.mat' ];
    
    if exist(pcl_file_name, 'file')
        disp(['skip ', pcl_file_name, '...']);
    else
        save(pcl_file_name, 'points', '-v6');
    end
    
    % add ground to the points
    fMinResolution = 0.02 * scale;
    
    x = -4:fMinResolution:4;
    y = -4:fMinResolution:4;
    
    [X,Y] = meshgrid(x,y);
    ground1 = [X(:), Y(:), min_p(3)*ones(length(X(:)),1)];
    
% %     x = -20:fMinResolution*10:20;
% %     y = -20:fMinResolution*10:20;
% %     
% %     [X,Y] = meshgrid(x,y);
% %     ground2 = [X(:), Y(:), -0.5*ones(length(X(:)),1)];
    
    points = [ points; ground1 ];
    
    x = -4:fMinResolution:4;
    y = -4:fMinResolution:4;
    z = -1:fMinResolution:4;
    
    [X,Z] = meshgrid(x,z);
    wall1 = [X(:), -4*ones(length(X(:)),1), Z(:)];
    wall2 = [X(:), 4*ones(length(X(:)),1), Z(:)];
    [Y,Z] = meshgrid(y,z);
    wall3 = [-4*ones(length(X(:)),1), Y(:), Z(:)];
    wall4 = [4*ones(length(X(:)),1), Y(:), Z(:)];
    
    points = [ points; wall1; wall2; wall3; wall4 ];
    %%
    
    Eul_x = [pi+pi/2, pi+pi/2-pi/6, pi+pi/2-2*pi/6]; % <sphere> Eul_x = [pi+pi/2+2*pi/6, pi+pi/2+pi/6, pi+pi/2, pi+pi/2-pi/6, pi+pi/2-2*pi/6];
    
    count = 0;
    
    for e_i = 1:length(Eul_x)
        
        interval = pi/180*36 * 1;
        
        eul_x = Eul_x(e_i);
        
        Eul_z = 0 : interval : pi/180*360-0.01;
        eul_y = 0;
        
        for e_j = 1:length(Eul_z)
            
            count = count + 1;
            
            file_name_save = [dest_folder, '/', object_name, '_' num2str(count), '.mat' ];
            
            if exist(file_name_save, 'file')
                disp(['skip ', file_name_save, '...']);
                load(file_name_save);
                depth_map = imresize(depth_map, [227, 227]);
                depth_map = single(depth_map);
                save(file_name_save, 'depth_map', 'eul', '-v6');
                continue;
            end
            
            eul_z = Eul_z(e_j);
            
            %% camera pose
            eul = [eul_z eul_y eul_x ];
            camera_R = eul2rotm(eul)';
            
            camera_dir = [0 0 1] * camera_R;
            origin = [0 0 0];
            
            sphere = [origin sphere_radius];
            lines = [origin camera_dir];
            
            inter_pts = intersectLineSphere(lines, sphere);
            
            angle = acos( dot(inter_pts(1,:), camera_dir) / (sqrt(inter_pts(1,:)*inter_pts(1,:)')*sqrt(camera_dir*camera_dir')) );
            
            if angle >= pi/2
                camera_pos = inter_pts(1,:);
            else
                camera_pos = inter_pts(2,:);
            end
            
            %camera_pos = [ 0 0 1 ]
            
            %% Hidden points removal
            radius = 2.5;
            
            visible_idx = HPR( points, camera_pos, radius );
            
            points = points*scale;
            %points = points + repmat(camera, [size(points,1), 1]);
            visualable_pts = points(visible_idx,:);
            hidden_pts = points;
            hidden_pts(visible_idx,:) = [];
            
            if flag
                figure(2)
                cam = plotCamera('Location', camera_pos, 'Orientation', camera_R, 'Opacity', 0, 'Size', 0.1);
                drawnow();
                hold on;
                xlim([-2 2]); ylim([-2 2]); zlim([-2 2]);
                
                figure(3)
                cla;
                xlim([-2 2]); ylim([-2 2]); zlim([-1 2]);
                plot3(hidden_pts(1:10:end,1),hidden_pts(1:10:end,2),hidden_pts(1:10:end,3),'r+')
                hold on
                plot3(visualable_pts(1:10:end,1),visualable_pts(1:10:end,2),visualable_pts(1:10:end,3),'b+')
                hold on
                cam = plotCamera('Location', camera_pos, 'Orientation', camera_R, 'Opacity', 0, 'Size', 0.2);
                drawnow();
            end
            
            %visualable_pts= points;
            %% project 3D point cloud to 2D depth map
            
            pts = (visualable_pts-repmat(camera_pos, [size(visualable_pts,1) 1]))*inv(camera_R);
            
            depth_map = getDethMap( zeros(500, 500), pts');
            
            for i = 1:Nfill
                [ depth_map ] = fill_hole( depth_map );
            end
            
            if flag             
                figure(4)
                cla
                axis([0 500 0 500]);
% %                 imagesc(depth_map);
                surf(depth_map(end:-1:1,:));
                view(2);
                camlight right;
                lighting phong;
                shading interp
                drawnow();
                pause(0.1);
            end
            
            save(file_name_save, 'depth_map', 'eul', '-v6');
            disp(['saving ', file_name_save, '...']);
        end
    end
    
end


return;
