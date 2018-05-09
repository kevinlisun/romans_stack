function HHA = depth2HHA(D, loc, C)
% function HHA = saveHHA(imName, C, outDir, D, RD)

C = [526.37013657, 0.00000000, 313.68782938; 0.00000000, 526.37013657, 259.01834898; 0.00000000, 0.00000000, 1.00000000];

% AUTORIGHTS
  
  D = double(D)./1000;
  missingMask = D == 0;
  [pc, N, yDir, h, pcRot, NRot] = processDepthImage(D*100, loc, missingMask, C);
  angl = acosd(min(1,max(-1,sum(bsxfun(@times, N, reshape(yDir, 1, 1, 3)), 3))));

  d = pc(:,:,3);
  maxd = min(max(max(d)), 90);
  mind = max(min(min(d)), 60);
 
  d = (d - mind) / (maxd - mind);
  I(:,:,1) = 255 - 255*d ;
  
  % Making the minimum depth to be 100, to prevent large values for disparity!!!
  %pc(:,:,3) = max(pc(:,:,3), 50); 
  %I(:,:,1) = 31000./pc(:,:,3);
  I(:,:,2) = h;
  I(:,:,3) = (angl+128-90); %Keeping some slack
  I = uint8(I);
  HHA = I;%(:,:,end:-1:1);
  
  HHA = interpolateImage(HHA);
end


function [pc, N, yDir, h, pcRot, NRot] = processDepthImage(z, loc, missingMask, C)
% function [pc, N, yDir, h, pcRot, NRot] = processDepthImage(z, missingMask, C)
% Input: 
%   z is in centimetres
%   C is the camera matrix

% AUTORIGHTS

  yDirParam.angleThresh = [45 15];
  yDirParam.iter = [5 5];
  yDirParam.y0 = [0 1 0]';

  normalParam.patchSize = [3 10];

  [X, Y, Z] = getPointCloudFromZ(z, loc, C, 1);
  pc = cat(3, X, Y, Z);

  % Compute the normals for this image
  [N1 b1] = computeNormalsSquareSupport(z./100, loc, missingMask, normalParam.patchSize(1),...
    1, C, ones(size(z)));
  [N2 b2] = computeNormalsSquareSupport(z./100, loc, missingMask, normalParam.patchSize(2),...
    1, C, ones(size(z)));
  % [N1 b1] = computeNormals2(pc(:,:,1), pc(:,:,2), pc(:,:,3), ones(size(pc(:,:,1))), normalParam.patchSize(1));
  % [N2 b2] = computeNormals2(pc(:,:,1), pc(:,:,2), pc(:,:,3), ones(size(pc(:,:,1))), normalParam.patchSize(2));
  
  N = N1; 

  % Compute the direction of gravity
  yDir = getYDir(N2, yDirParam);
  y0 = [0 1 0]';
  R = getRMatrix(y0, yDir);
  NRot = rotatePC(N, R');
  pcRot = rotatePC(pc, R');
  h = -pcRot(:,:,2);
  yMin = prctile(h(:), 0); if(yMin > -90) yMin = -130; end
  h = h-yMin;
end