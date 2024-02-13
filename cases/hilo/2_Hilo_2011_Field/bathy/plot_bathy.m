clear all
clf

% This script will load the Hilo Harbor bathy data, in lat/long

load hilo_grid_1_3_arcsec.txt
data=hilo_grid_1_3_arcsec;
xv=data(:,1);
yv=data(:,2);
zv=data(:,3);

mn=length(xv);
m=701; % taken from file
n=mn/m;

x=xv(1:m);
z=zeros(n,m);

count=0;
for j=1:n
    for i=1:m
        count=count+1;
        z(j,i)=-zv(count);
        if z(j,i)<=-30
            z(j,i)=-30;
        end
        if i==1
            y(j)=yv(count);
        end
    end
end
z=flipud(z);

pcolor(x,y,z)
shading interp
xlabel({'Longitude (degrees))'})
ylabel('Latitude (degrees)')
colormap(jet)
colorbar
axis equal
axis tight
%caxis([-30 0])
view(0,90)
hold on
plot(360-155.07,19.7576,'w.','MarkerSize',20)  % simulation control point
plot(360-(155+4.919/60),19+44.710/60,'k.','MarkerSize',20)  % HAI1125
plot(360-(155+4.198/60),19+44.500/60,'k.','MarkerSize',20)  % HAI1126
plot(360-(155.0553), 19.7308,'b.','MarkerSize',20)  % Tide Gage


print -djpeg100 bathy.jpg 




