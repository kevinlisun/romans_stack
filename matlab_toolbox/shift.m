function [img2] = shift(img, shift_x, shift_y)

    img2 = zeros(size(img,1), size(img,2));
    
    if shift_y <= 0
        img2(1:end+shift_y,:) = img(-shift_y+1:end,:);
    else
        img2(shift_y+1:end,:) = img(1:end-shift_y,:);
    end
    
    img = img2;
    img2 = zeros(size(img,1), size(img,2));
    
    if shift_x <= 0
        img2(:,1:end+shift_x) = img(:,-shift_x+1:end);
    else
        img2(:,shift_x+1:end) = img(:,1:end-shift_x);
    end