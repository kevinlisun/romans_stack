function img = interpolateImage2D(I)


    [row, col] = size(I);
    
    if col > row
        firstline = I(1, :);
        lastline = I(end, :);
        
        img = zeros(col, col);
        
        nup = fix((col - row)/2);
        ndown = col - row - nup;
        
        img(1:nup, :) = repmat(firstline, [nup, 1]);
        img(nup+1:nup+row, :) = I;
        img(end-ndown+1:end, :) = repmat(lastline, [ndown, 1]);
    elseif col < row
        firstline = I(:, 1);
        lastline = I(:, end);
        
        img = zeros(row, row);
        
        nleft = fix((row - col)/2);
        nright = row - col - nleft;
        
        img(:, 1:nleft) = repmat(firstline, [1, nleft]);
        img(:, nleft+1:nleft+col) = I;
        img(:, end-nright+1:end) = repmat(lastline, [1, nright]);
    else
        img = I;
    end
    
    img = single(img);
    