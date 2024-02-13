% loads and plots adcp data

clear all
clf

load TG_1617760_detided.txt
t=TG_1617760_detided(:,1);
TG_tsunami=TG_1617760_detided(:,2);

plot(t/3600,TG_tsunami,'.-b','linewidth',1)
xlabel('time after EQ (hrs)')
ylabel('Ocean Surface Elevation (m/s)')
title(['De-tided Tide Gage Data, Hilo Harbor'])
grid on
axis([7 13 -Inf Inf])

print -djpeg100 TideGage.jpg 



