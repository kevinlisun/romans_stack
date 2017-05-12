function label = findLabel(folder_name)

switch folder_name
    case 'airplane'
        label = 0;
    case 'bathtub'
        label = 1;
    case 'bed'
        label = 2;
    case 'bench'
        label = 3;
    case 'bookshelf'
        label = 4;
    case 'bottle'
        label = 5;
    case 'bowl'
        label = 6;
    case 'car'
        label = 7;
    case 'chair'
        label = 8;
    case 'cone'
        label = 9;
    case 'cup'
        label = 10;
    case 'curtain'
        label = 11;
    case 'desk'
        label = 12;
    case 'door'
        label = 13;
    case 'dresser'
        label = 14;
    case 'flower_pot'
        label = 15;
    case 'glass_box'
        label = 16;
    case 'guitar'
        label = 17;
    case 'keyboard'
        label = 18;
    case 'lamp'
        label = 19;
    case 'laptop'
        label = 20;
    case 'mantel'
        label = 21;
    case 'monitor'
        label = 22;
    case 'night_stand'
        label = 23;
    case 'person'
        label = 24;
    case 'piano'
        label = 25;
    case 'plant'
        label = 26;
    case 'radio'
        label = 27;
    case 'range_hood'
        label = 28;
    case 'sink'
        label = 29;
    case 'sofa'
        label = 30;
    case 'stairs'
        label = 31;
    case 'stool'
        label = 32;
    case 'table'
        label = 33;
    case 'tent'
        label = 34;
    case 'toilet'
        label = 35;
    case 'tv_stand'
        label = 36;
    case 'vase'
        label = 37;
    case 'wardrobe'
        label = 38;
    case 'xbox'
        label = 39;
    otherwise
        disp('ERROR: No such category');
        disp(folder_name);
        label = -1;
end