function [t,x,y,z] = loadAndProcess(directory)

    if nargin < 1
            directory = "./";
    end
    directory = string(directory) + "/";

    % Load the data.
    y = readmatrix(directory + "y.csv");
    z = readmatrix(directory + "z.csv");
    t = readmatrix(directory + "t.csv");

    % Interpolate y and z onto a uniform t grid of 100 frames per 2*pi.
    ts = linspace(t(1),t(end),round(100*(t(end)-t(1))/(2*pi)));
    y = interp1(t,y',ts)';
    z = interp1(t,z',ts)';
    t = ts;

    % Build the spatial domain.
    x = linspace(0,1,size(y,1))';
end