function [v1 f1] = subdivide_tri( xyz, faces )
%
% Vectorized Triangle Subdivision: split each triangle 
% input face into four new triangles. 
% 
% usage: [v1 f1] = subdivide( xyz, faces );
%
%  author:  Peter A. Karasev     25 Nov 2009


numverts = size(xyz,1);
numfaces = size(faces,1);
disp(['Input mesh: ' num2str(numfaces) ' triangles, ' ... 
    num2str(numverts) ' vertices.']);

fk1 = faces(:,1);
fk2 = faces(:,2);
fk3 = faces(:,3);

% create averages of pairs of vertices (k1,k2), (k2,k3), (k3,k1)
    m1x = (xyz( fk1,1) + xyz( fk2,1) )/2;
    m1y = (xyz( fk1,2) + xyz( fk2,2) )/2;
    m1z = (xyz( fk1,3) + xyz( fk2,3) )/2;
    
    m2x = (xyz( fk2,1) + xyz( fk3,1) )/2;
    m2y = (xyz( fk2,2) + xyz( fk3,2) )/2;
    m2z = (xyz( fk2,3) + xyz( fk3,3) )/2;
    
    m3x = (xyz( fk3,1) + xyz( fk1,1) )/2;
    m3y = (xyz( fk3,2) + xyz( fk1,2) )/2;
    m3z = (xyz( fk3,3) + xyz( fk1,3) )/2;

    
vnew = [ [m1x m1y m1z]; [m2x m2y m2z]; [m3x m3y m3z] ];
clear m1x m1y m1z m2x m2y m2z m3x m3y m3z
[vnew_ ii jj] = unique(vnew, 'rows' );

clear vnew; 
m1 = jj(1:numfaces)+numverts;
m2 = jj(numfaces+1:2*numfaces)+numverts;
m3 = jj(2*numfaces+1:3*numfaces)+numverts;

tri1 = [fk1 m1 m3];
tri2 = [fk2 m2 m1];
tri3 = [ m1 m2 m3];
tri4 = [m2 fk3 m3];
clear m1 m2 m3 fk1 fk2 fk3
 
v1 = [xyz; vnew_]; % the new vertices
f1 = [tri1; tri2; tri3; tri4]; % the new faces
disp(['Output mesh: ' num2str(size(f1,1)) ' triangles, ' ... 
    num2str(size(v1,1))  ' vertices.']);