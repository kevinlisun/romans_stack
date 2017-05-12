function depth_map = getDethMap( img, UV )

% UV is 3 by N matrix [u; v; d]

    height = size(img, 1);
    width = size(img, 2);
    
    cx = width/2;
    cy = height/2;
    
    fx = 500;
    fy = 500;
    
    depth_map = img;
    
    for i = 1:size(UV,2)
        x = UV(1, i);
        y = UV(2, i);
        z = UV(3, i);
        
        x0 = round( fx*x/z + cx );
        y0 = round( fy*y/z + cy );
        
        
        if x0>=1 & x0<=width & y0>=1 & y0<=height
            if depth_map(y0,x0)~=0
                depth_map(y0,x0) = min(z,depth_map(y0,x0));
            else
                depth_map(y0,x0) = z;
            end
        end
    end
    
    depth_map(isnan(depth_map)) = 0;
    depth_map(depth_map<0) = 0;
        