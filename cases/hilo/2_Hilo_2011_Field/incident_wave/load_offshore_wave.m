clear all
clf
% 
% load ts_hilo_most.txt  % MOST simulation control point, at -155.07,19.7576
% t=ts_hilo_most(:,1);
% z=ts_hilo_most(:,2);  %tsunami only

load se.dat  % NEOWAVE simulation control point, at -155.07,19.7576
t2=se(:,1)/60;
z2=se(:,2)-se(1,2);  %tsunami only

% load gauge3333.txt   % GEOCLAW simulation control point, at -155.07,19.7576 
% t3=gauge3333(:,1);
% z3=gauge3333(:,3);  %tsunami only

plot(t2,z2)
ylabel('Ocean Surface Elevation at Control Point (m)')

axis([7 11 -Inf Inf])

xlabel('Time post-EQ (hrs)')
print -djpeg100 incident_wave.jpg






