function plot_topo(topo, eta, filename)

xmax = 70;
ymax = 30;

[X, Y] = meshgrid(0:xmax, 0:ymax);

Z   = topo(X, Y);
ETA = zeros(ymax+1, xmax+1) + eta;

surf(X, Y, Z, ...
    'FaceColor', '#B1362A' ...
    );

view([300, 5]);
pbaspect([xmax/ymax 1 1]);
hold on
grid off

surf(X, Y, ETA, ...
    'FaceAlpha', 0.5, ...
    'FaceColor', '#599DEE', ...
    'EdgeColor','none' ...
    );

xlabel('{\it x} (m)');
ylabel('{\it y} (m)');

xlim([0, xmax]);
ylim([0, ymax]);

hold off

filename_full = strcat(fullfile('results', filename), '.pdf');

if not( isfile(filename_full) )
    exportgraphics(gcf, filename_full, 'Resolution', 600)
end

end