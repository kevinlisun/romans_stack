function depth_map = gridDethMap( img, UV )

% UV is 3 by N matrix [u; v; d]

    height = size(img, 1);
    width = size(img, 2);
    
    cx = width/2;
    cy = height/2;
    
    fx = 500;
    fy = 500;
    
    depth_map = img;
    
    X = [];
    Y = [];
    Z = [];
    
    for i = 1:size(UV,2)
        x = UV(1, i);
        y = UV(2, i);
        z = UV(3, i);
        
        x0 = fx*x/z + cx;
        y0 = fy*y/z + cy;
    
        
        if x0>=1 & x0<=width & y0>=1 & y0<=height
            X = [ X; x0 ];
            Y = [ Y; y0 ];
            Z = [ Z; z ];
        end
    end
      
    dx = 1:1:width;
    dy = 1:1:height;
    [xq,yq] = meshgrid(dx, dy);
    depth_map = griddata(X,Y,Z,xq,yq, 'natural');