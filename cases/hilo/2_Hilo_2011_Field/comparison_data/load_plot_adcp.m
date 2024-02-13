% loads and plots adcp data

clear all
clc

adcp_loc = {'1125','1126'}; % entrance: adcp @ hilo harbor entrance, harbor: adcp inside the hilo harbor

for n=1:length(adcp_loc)
    
    data = load (['HAI' char(adcp_loc(n)) '_detided_harmonic.txt']);
    t=data(:,1);
    e_w=data(:,2);
    n_s=data(:,3)
    
    figure (n)
    clf
    if n==1
        loc='HA 1125: Approach to Hilo Harbor';
        loc2='HA_1125_Approach_to_Hilo_Harbor';
    end
    
    if n==2
        loc='HA 1126:Hilo Harbor';
        loc2='HA_1126_Hilo_Harbor';
    end
    
    subplot(2,1,1)
    plot(t,e_w,'.-b','linewidth',1)
    xlabel('time after EQ (hrs)')
    ylabel('u (cm/s)')
    title([ loc ' - E-W Current Speeds'])
    grid on 
    axis([7 13 -Inf Inf])
    
    subplot(2,1,2)
    plot(t,n_s,'.-g','linewidth',1)
    xlabel('time after EQ (hrs)')
    ylabel('v (cm/s)')
    title([ loc ' - N-S Current Speeds'])
    grid on
    axis([7 13 -Inf Inf])
    
    eval(['print -djpeg100 '  loc2])
end



