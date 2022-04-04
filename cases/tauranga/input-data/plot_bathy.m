clear all
clf

% This script will load the Tauranga Bay bathy data, generate coordinates in
% cartesian coordinates, and rotate the grid such that the domain is
% rectangular with axis aligned with x and y. If you want to load the data
% in true (lat,long) coordinates, take a look at the file
% TAU_Whole_Harbour_10_m_srf6.txt


% work below commented out - skip to loading the generated data.  Uncomment
% to re-generate (takes about 10 minutes).

% load TAU_Whole_Harbour_10_m_srf6_hmat.txt -ascii
%  
% ht = flipud(TAU_Whole_Harbour_10_m_srf6_hmat);
%     
% [m, n] = size(ht);
%  
% dx = 10;
% dy = 10;
% 
% x = ht * 0.;
% 
% for i = 1:n
%    x(:,i) = x(:,i) + (i-1) * dx;
% end
% 
% y = ht * 0.;
% 
% for j = 1:m
%    y(j,:) = y(j,:) + (j-1) * dy;
% end
% 
% ang = 49;
% R = [ cosd(ang), -sind(ang); sind(ang), cosd(ang) ]; % rotation matrix
% 
% xr = x * 0;
% yr = y * 0;
% 
% for j = 1:m
%     for i = 1:n  % rotate coordinates "ang"
%         xr(j,i) = x(j,i) * R(1,1) + y(j,i) * R(1,2);
%         yr(j,i) = x(j,i) * R(2,1) + y(j,i) * R(2,2);
%     end
% end
% 
% xc = reshape( xr, [m*n,1] );
% yc = reshape( yr, [m*n,1] );
% hc = reshape( ht, [m*n,1] );
% 
% disp('TriScatteredInterp')
% F = TriScatteredInterp(xc, yc, hc, 'natural');
% 
% xn = [-2.2e4:dx:1.90e4];
% yn = [ 1.0e4:dx:3.24e4];
% 
% nx = length(xn);
% ny = length(yn);
% 
% for i = 1:nx
%     [i nx]
%     for j = 1:ny
%         hm(j,i) = F( xn(i), yn(j) );
% 
%         if yn(j) < 2.12e4 & isnan( hm(j,i) )
%             hm(j,i) = 9999;
%         end
%     end
% end
% 
% 
% for j = 1:ny
% 
%     if isnan( hm(j,nx) ) % for left corner
%         for i = 2:nx
%             if isnan( hm(j,i) )
%                 hm(j,i) = hm(j,i-1);
%             end
%         end
%     end
% 
%     if isnan( hm(j,1) ) % for right corner
%         for i = nx:-1:1
%             if isnan( hm(j,i) )
%                 hm(j,i) = hm(j,i+1);
%             end
%         end
%     end
% 
% end
% 
% save TAU_Whole_Harbour_10_m_srf6_hmat.mat

load TAU_Whole_Harbour_10_m_srf6_hmat.mat

x = xn - xn(1);
y = yn - yn(1);
z = -hm;

writematrix(z, 'bathymetry.csv')

% stage points
AB   = [2.724e4, 1.846e4];
Tug  = [3.085e4, 1.512e4];
SP   = [3.200e4, 1.347e4];
Mot  = [3.005e4, 1.610e4];
ADCP = [2.925e4, 1.466e4];

pcolor(x,y,z)
shading interp
xlabel('Longshore Coordinate (m)')
ylabel('Cross-shore Coordinate (m)')
colormap(jet)
colorbar
axis equal
axis tight
caxis([-25 35])
view(0,90)

hold on

plot(AB(1),   AB(2),   'w*')
plot(Tug(1),  Tug(2),  'w*')
plot(SP(1),   SP(2),   'w*')
plot(Mot(1),  Mot(2),  'w*')
plot(ADCP(1), ADCP(2), 'ws')