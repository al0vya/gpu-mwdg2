clf
clear all

fid(1)=fopen('seaside_matrix.bin', 'r');
dum=fread(fid(1),1,'int32');
m=fread(fid(1),1,'int32');
n=fread(fid(1),1,'int32');
x=fread(fid(1),m,'float32');
y=fread(fid(1),n,'float32');
z=fread(fid(1),[m,n],'float32');
fclose all;

dx=x(2)-x(1);
xmax=x(m);
xmin=x(1);

m_new=round( (xmax+0)/dx );

x_new(m_new)=xmax;
for i=m_new-1:-1:1
   x_new(i)=x_new(i+1)-dx;
end

h_new=zeros(m_new,n);
count=0;
for i=m_new:-1:1
   if x_new(i)>=32.5
      count=count+1;
      h_new(i,:)=z(m-count+1,:);  % lidar data is only on the dry portion, where the model is
   elseif x_new(i)>=17.5
      h_new(i,:)=h_new(i+1,:)-dx/30;     
   elseif x_new(i)>=10.
      h_new(i,:)=h_new(i+1,:)-dx/15;
   else
      h_new(i,:)=0.;   
   end
end
   
z=h_new';
x=x_new';
y=y;

%% plot the data

pcolor(x,y,z)
shading interp
text(0,-2,0.5,['WAVEMAKER'],'VerticalAlignment','cap','HorizontalAlignment','center','Rotation',90,'FontSize',10,'Color','w')
ylabel({'Longshore Location (m)'})
xlabel('Cross-shore Location from Wavemaker (m)')
colormap(jet)
colorbar
axis equal
axis tight
view(0,90)

print -djpeg100 bathy.jpg 