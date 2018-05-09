function result = is_in_set(inst, set)

    result = false;
    for i = 1:length(set)
        if strcmp(inst, set{i})
            result = true;
            return;
        end
    end