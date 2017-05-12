function label = findLabel3(folder_name)


switch folder_name
    case 'apple'
        label = 0;
    case 'ball'
        label = 1;
    case 'banana'
        label = 2;
    case 'bell_pepper'
        label = 3;
    case 'binder'
        label = 4;
    case 'bowl'
        label = 5;
    case 'calculator'
        label = 6;
    case 'camera'
        label = 7;
    case 'cap'
        label = 8;
    case 'cell_phone'
        label = 9;
    case 'cereal_box'
        label = 10;
    case 'coffee_mug'
        label = 11;
    case 'comb'
        label = 12;
    case 'dry_battery'
        label = 13;
    case 'flashlight'
        label = 14;
    case 'food_bag'
        label = 15;
    case 'food_box'
        label = 16;
    case 'food_can'
        label = 17;
    case 'food_cup'
        label = 18;
    case 'food_jar'
        label = 19;
    case 'garlic'
        label = 20;
    case 'glue_stick'
        label = 21;
    case 'greens'
        label = 22;
    case 'hand_towel'
        label = 23;
    case 'instant_noodles'
        label = 24;
    case 'keyboard'
        label = 25;
    case 'kleenex'
        label = 26;
    case 'lemon'
        label = 27;
    case 'lightbulb'
        label = 28;
    case 'lime'
        label = 29;
    case 'marker'
        label = 30;
    case 'mushroom'
        label = 31;
    case 'notebook'
        label = 32;
    case 'onion'
        label = 33;
    case 'orange'
        label = 34;
    case 'peach'
        label = 35;
    case 'pear'
        label = 36;
    case 'pitcher'
        label = 37;
    case 'plate'
        label = 38;
    case 'pliers'
        label = 39;
    case 'potato'
        label = 40;
    case 'rubber_eraser'
        label = 41;
    case 'scissors'
        label = 42;
    case 'shampoo'
        label = 43;
    case 'soda_can'
        label = 44;
    case 'sponge'
        label = 45;
    case 'stapler'
        label = 46;
    case 'tomato'
        label = 47;
    case 'toothbrush'
        label = 48;
    case 'toothpaste'
        label = 49;
    case 'water_bottle'
        label = 50;
    otherwise
        disp('ERROR: No such category');
        disp(folder_name);
        label = -1;
end