function [is_line] = is_in_the_line(x, y, theta, rho)

    theta = theta / 180 * pi;
    
    thres = 1;
    
    abs(rho - (cos(theta)*x + sin(theta)*y));
    
    if abs(rho - (cos(theta)*x + sin(theta)*y)) <= thres
        is_line = true;
    else
        is_line = false;
        
    end
end