function [ img_out ] = fill_hole( img )
    
    h = size(img,1);
    w = size(img,2);
    
    img_n = [ img(2:end,:); zeros(1,w) ];
    img_s = [ zeros(1,w); img(1:end-1,:) ];
    img_e = [ zeros(h,1), img(:,1:end-1) ];
    img_w = [ img(:,2:end), zeros(h,1) ];
    
    img_ne = [ zeros(h,1), [img(2:end,1:end-1);zeros(1,w-1)] ];
    img_nw = [ [img(2:end,2:end);zeros(1,w-1)], zeros(h,1) ];
    img_se = [ zeros(h,1), [zeros(1,w-1);img(1:end-1,1:end-1)] ];
    img_sw = [ [zeros(1,w-1); img(1:end-1,2:end)], zeros(h,1) ];
    
    imgs = zeros(h,w,9);
    imgs(:,:,1) = img_n;
    imgs(:,:,2) = img_s;
    imgs(:,:,3) = img_e;
    imgs(:,:,4) = img_w;
    imgs(:,:,5) = img_ne;
    imgs(:,:,6) = img_nw;
    imgs(:,:,7) = img_se;
    imgs(:,:,8) = img_sw;
    imgs(:,:,9) = img;
    
    imgs(imgs==0) = NaN;
    
    img_out = nanmedian(imgs,3);
    
    img_out(isnan(img_out)) = 0;
    img_out(img_out<0) = 0;
    
    img_out(img~=0) = 0;
    img_out = img_out + img;