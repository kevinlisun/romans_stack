function label = findLabel2(folder_name)

switch folder_name
    case 'background'
        label = 0;
    case 'bottles'
        label = 1;
    case 'cans'
        label = 2;
    case 'chains'
        label = 3;
    case 'cloth'
        label = 4;
    case 'gloves'
        label = 5;
    case 'metal_objects'
        label = 6;
    case 'pipe_joints'
        label = 7;
    case 'plastic_pipes'
        label = 8;
    case 'sponges'
        label = 9;
    case 'wood_blocks'
        label = 10;
    otherwise
        disp('ERROR: No such category');
        disp(folder_name);
        label = -1;
end