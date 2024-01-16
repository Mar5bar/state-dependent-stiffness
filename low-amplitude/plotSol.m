function plotSol(directory, subMean)

    if nargin < 1
            directory = "./";
    end
    if nargin < 2
        subMean = false;
    end

    skip = 4;

    [t,x,y,z] = loadAndProcess(directory);


    m = t > t(end) - 2*pi;
    y = y (:,m);
    y = y(:,1:skip:end);

    if subMean
        y = y - (max(y,[],2) + min(y,[],2))/2;
    end

    ylims = [min(min(y)), max(max(y))];
    ylims(1) = ylims(1) - 0.1*max(abs(ylims));
    ylims(2) = ylims(2) + 0.1*max(abs(ylims));

    nexttile()
    plot(x,y,'black')
    ylim(ylims)
    xlabel('$x$')
    if subMean
        ylabel('$y - y_c$')
    else
        ylabel('$y$')
    end

end