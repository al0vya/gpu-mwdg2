# This script is used to postprocess results and plot graphs for the Hilo harbour
# test case at coastal.usc.edu/currents_workshop/problems/prob2.html

import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
    
def plot_predictions():
    print('Plotting predictions...')
    
    control_point_data = np.loadtxt( fname=os.path.join('input-data', 'se.txt') )
    time_control_point = control_point_data[:,0] / 60
    eta_control_point_no_tide = control_point_data[:,1]-control_point_data[0,1]
    
    tide_gauge_data = np.loadtxt( fname=os.path.join('input-data', 'TG_1617760_detided.txt') )
    time_tide_gauge = tide_gauge_data[:,0] / 3600
    eta_tide_gauge  = tide_gauge_data[:,1]
    
    timeshift = time_control_point[0]
    
    eta_predictions_0    = np.loadtxt(fname=os.path.join('eps-0',    'res.stage'), skiprows=11, delimiter=' ')
    eta_predictions_1e_3 = np.loadtxt(fname=os.path.join('eps-1e-3', 'res.stage'), skiprows=11, delimiter=' ')
    eta_predictions_1e_4 = np.loadtxt(fname=os.path.join('eps-1e-4', 'res.stage'), skiprows=11, delimiter=' ')
    
    control_point_0    = eta_predictions_0[:,2]    - 30
    control_point_1e_3 = eta_predictions_1e_3[:,2] - 30
    control_point_1e_4 = eta_predictions_1e_4[:,2] - 30
    
    tide_gauge_0    = eta_predictions_0[:,3]    + 19.546045 - 30
    tide_gauge_1e_3 = eta_predictions_1e_3[:,3] + 19.546045 - 30
    tide_gauge_1e_4 = eta_predictions_1e_4[:,3] + 19.546045 - 30
    
    vel_ha1125 = np.loadtxt( fname=os.path.join('input-data', 'HAI1125_detided_harmonic.txt') )
    vel_ha1126 = np.loadtxt( fname=os.path.join('input-data', 'HAI1126_detided_harmonic.txt') )
    
    ha1125_vy_0    = np.loadtxt(fname=os.path.join('eps-0',    'res.yvelocity'), skiprows=11, delimiter=' ')[:,4]
    ha1125_vy_1e_3 = np.loadtxt(fname=os.path.join('eps-1e-3', 'res.yvelocity'), skiprows=11, delimiter=' ')[:,4]
    ha1125_vy_1e_4 = np.loadtxt(fname=os.path.join('eps-1e-4', 'res.yvelocity'), skiprows=11, delimiter=' ')[:,4]
    
    ha1126_vx_0    = np.loadtxt(fname=os.path.join('eps-0',    'res.xvelocity'), skiprows=11, delimiter=' ')[:,5]
    ha1126_vx_1e_3 = np.loadtxt(fname=os.path.join('eps-1e-3', 'res.xvelocity'), skiprows=11, delimiter=' ')[:,5]
    ha1126_vx_1e_4 = np.loadtxt(fname=os.path.join('eps-1e-4', 'res.xvelocity'), skiprows=11, delimiter=' ')[:,5]
    
    map_el_0    = np.loadtxt(fname=os.path.join('eps-0',    'res-1.elev'), skiprows=6)
    map_el_1e_3 = np.loadtxt(fname=os.path.join('eps-1e-3', 'res-1.elev'), skiprows=6)
    map_el_1e_4 = np.loadtxt(fname=os.path.join('eps-1e-4', 'res-1.elev'), skiprows=6)
    
    map_vx_0    = np.loadtxt(fname=os.path.join('eps-0',    'res-1.vx'), skiprows=6)
    map_vx_1e_3 = np.loadtxt(fname=os.path.join('eps-1e-3', 'res-1.vx'), skiprows=6)
    map_vx_1e_4 = np.loadtxt(fname=os.path.join('eps-1e-4', 'res-1.vx'), skiprows=6)
    
    map_vy_0    = np.loadtxt(fname=os.path.join('eps-0',    'res-1.vy'), skiprows=6)
    map_vy_1e_3 = np.loadtxt(fname=os.path.join('eps-1e-3', 'res-1.vy'), skiprows=6)
    map_vy_1e_4 = np.loadtxt(fname=os.path.join('eps-1e-4', 'res-1.vy'), skiprows=6)
        
    time = eta_predictions_1e_3[:,0]/3600 + timeshift
    
    timeslice = (time > 8.5) & (time < 11)
    
    rec_dd = lambda: collections.defaultdict(rec_dd)
    
    table_data = rec_dd()

    table_data[1e-3]['cp']['RMSE'] = np.sqrt( np.square( control_point_1e_3[timeslice] - control_point_0[timeslice]).mean() )
    table_data[1e-4]['cp']['RMSE'] = np.sqrt( np.square( control_point_1e_4[timeslice] - control_point_0[timeslice]).mean() )
    table_data[1e-3]['tg']['RMSE'] = np.sqrt( np.square( tide_gauge_1e_3[timeslice]    - tide_gauge_0[timeslice]   ).mean() )
    table_data[1e-4]['tg']['RMSE'] = np.sqrt( np.square( tide_gauge_1e_4[timeslice]    - tide_gauge_0[timeslice]   ).mean() )
    table_data[1e-3]['25']['RMSE'] = np.sqrt( np.square( ha1125_vy_1e_3[timeslice]     - ha1125_vy_0[timeslice]    ).mean() )
    table_data[1e-4]['25']['RMSE'] = np.sqrt( np.square( ha1125_vy_1e_4[timeslice]     - ha1125_vy_0[timeslice]    ).mean() )
    table_data[1e-3]['26']['RMSE'] = np.sqrt( np.square( ha1126_vx_1e_3[timeslice]     - ha1126_vx_0[timeslice]    ).mean() )
    table_data[1e-4]['26']['RMSE'] = np.sqrt( np.square( ha1126_vx_1e_4[timeslice]     - ha1126_vx_0[timeslice]    ).mean() )
    table_data[1e-3]['el']['RMSE'] = np.sqrt( np.square( map_el_1e_3                   - map_el_0                  ).mean() )
    table_data[1e-4]['el']['RMSE'] = np.sqrt( np.square( map_el_1e_4                   - map_el_0                  ).mean() )
    table_data[1e-3]['vx']['RMSE'] = np.sqrt( np.square( map_vx_1e_3                   - map_vx_0                  ).mean() )
    table_data[1e-4]['vx']['RMSE'] = np.sqrt( np.square( map_vx_1e_4                   - map_vx_0                  ).mean() )
    table_data[1e-3]['vy']['RMSE'] = np.sqrt( np.square( map_vy_1e_3                   - map_vy_0                  ).mean() )
    table_data[1e-4]['vy']['RMSE'] = np.sqrt( np.square( map_vy_1e_4                   - map_vy_0                  ).mean() )
    
    table_data[1e-3]['cp']['corr'] = np.corrcoef( x=control_point_1e_3[timeslice], y=control_point_0[timeslice] )[0][1]
    table_data[1e-4]['cp']['corr'] = np.corrcoef( x=control_point_1e_4[timeslice], y=control_point_0[timeslice] )[0][1]
    table_data[1e-3]['tg']['corr'] = np.corrcoef( x=tide_gauge_1e_3[timeslice]   , y=tide_gauge_0[timeslice]    )[0][1]
    table_data[1e-4]['tg']['corr'] = np.corrcoef( x=tide_gauge_1e_4[timeslice]   , y=tide_gauge_0[timeslice]    )[0][1]
    table_data[1e-3]['25']['corr'] = np.corrcoef( x=ha1125_vy_1e_3[timeslice]    , y=ha1125_vy_0[timeslice]     )[0][1]
    table_data[1e-4]['25']['corr'] = np.corrcoef( x=ha1125_vy_1e_4[timeslice]    , y=ha1125_vy_0[timeslice]     )[0][1]
    table_data[1e-3]['26']['corr'] = np.corrcoef( x=ha1126_vx_1e_3[timeslice]    , y=ha1126_vx_0[timeslice]     )[0][1]
    table_data[1e-4]['26']['corr'] = np.corrcoef( x=ha1126_vx_1e_4[timeslice]    , y=ha1126_vx_0[timeslice]     )[0][1]        
    table_data[1e-3]['el']['corr'] = np.corrcoef( x=map_el_1e_3.flatten()        , y=map_el_0.flatten()         )[0][1]        
    table_data[1e-4]['el']['corr'] = np.corrcoef( x=map_el_1e_4.flatten()        , y=map_el_0.flatten()         )[0][1]        
    table_data[1e-3]['vx']['corr'] = np.corrcoef( x=map_vx_1e_3.flatten()        , y=map_vx_0.flatten()         )[0][1]        
    table_data[1e-4]['vx']['corr'] = np.corrcoef( x=map_vx_1e_4.flatten()        , y=map_vx_0.flatten()         )[0][1]        
    table_data[1e-3]['vy']['corr'] = np.corrcoef( x=map_vy_1e_3.flatten()        , y=map_vy_0.flatten()         )[0][1]        
    table_data[1e-4]['vy']['corr'] = np.corrcoef( x=map_vy_1e_4.flatten()        , y=map_vy_0.flatten()         )[0][1]        
    
    table = [
        ('h + z', table_data[1e-3]['cp']['RMSE'], table_data[1e-4]['cp']['RMSE'], table_data[1e-3]['cp']['corr'], table_data[1e-4]['cp']['corr']),
        ('h + z', table_data[1e-3]['tg']['RMSE'], table_data[1e-4]['tg']['RMSE'], table_data[1e-3]['tg']['corr'], table_data[1e-4]['tg']['corr']),
        ('vy',    table_data[1e-3]['25']['RMSE'], table_data[1e-4]['25']['RMSE'], table_data[1e-3]['25']['corr'], table_data[1e-4]['25']['corr']),
        ('vx',    table_data[1e-3]['26']['RMSE'], table_data[1e-4]['26']['RMSE'], table_data[1e-3]['26']['corr'], table_data[1e-4]['26']['corr']),
        ('h + z', table_data[1e-3]['el']['RMSE'], table_data[1e-4]['el']['RMSE'], table_data[1e-3]['el']['corr'], table_data[1e-4]['el']['corr']),
        ('vx',    table_data[1e-3]['vx']['RMSE'], table_data[1e-4]['vx']['RMSE'], table_data[1e-3]['vx']['corr'], table_data[1e-4]['vx']['corr']),
        ('vy',    table_data[1e-3]['vy']['RMSE'], table_data[1e-4]['vy']['RMSE'], table_data[1e-3]['vy']['corr'], table_data[1e-4]['vy']['corr'])
    ]
    
    table_df = pd.DataFrame(
        data=table,
        index=['Control point', 'Tide gauge', 'ADCP HA1125', 'ADCP HA1126', 'Spatial map at ts', 'Spatial map at ts', 'Spatial map at ts'],
        columns=['Quantity', 'RMSE, \epsilon = 10-3', 'RMSE, \epsilon = 10-4', 'R2, \epsilon = 10-3', 'R2, \epsilon = 10-4']
    )
    
    table_df.to_csv('table.csv')
    
    fig, axs = plt.subplots(
        figsize=(6,4),
        nrows=2,
        ncols=2,
        sharex=True
    )
    
    fig.subplots_adjust(
        hspace=0.3,
        wspace=0.35
    )
    
    lines = []
    
    lines.append( axs[0,0].plot(time, control_point_1e_3, label='GPU-MWDG2, $\epsilon = 10^{-3}$')[0] )
    lines.append( axs[0,0].plot(time, control_point_1e_4, label='GPU-MWDG2, $\epsilon = 10^{-4}$')[0] )
    lines.append( axs[0,0].plot(time, control_point_0,    label='GPU-DG2')                        [0] )
    
    lines.append( axs[0,0].plot(time_control_point, eta_control_point_no_tide, label='Experimental', color='k')[0] )
    
    axs[0,0].set_xlim( (8.5,11) )
    axs[0,0].set_ylim( (-1.5,1.7) )
    
    axs[0,0].set_title('Control point', fontsize='medium')
    axs[0,0].set_ylabel('$h + z$ (m)')
    
    axs[0,1].plot(time, tide_gauge_1e_3)
    axs[0,1].plot(time, tide_gauge_1e_4)
    axs[0,1].plot(time, tide_gauge_0)
    
    axs[0,1].plot(time_tide_gauge, eta_tide_gauge, label='Experimental', color='k')
    
    axs[0,1].set_xlim( (8.5,11) )
    axs[0,1].set_ylim( (-2.5,2.5) )
    
    axs[0,1].set_title('Tide gauge', fontsize='medium')
    axs[0,1].set_ylabel('$h + z$ (m)')
    
    axs[1,0].plot(time, ha1125_vy_1e_3)
    axs[1,0].plot(time, ha1125_vy_1e_4)
    axs[1,0].plot(time, ha1125_vy_0)
    
    axs[1,0].plot(vel_ha1125[:,0], vel_ha1125[:,2] / 100, label='Experimental', color='k')
    
    axs[1,0].set_xlim( (8.5,11) )
    axs[1,0].set_ylim( (-1.4,1.6) )
    
    axs[1,0].set_title('ADCP HA1125', fontsize='medium')
    axs[1,0].set_ylabel('$v$ (m/s)')
    
    axs[1,1].plot(time, ha1126_vx_1e_3)
    axs[1,1].plot(time, ha1126_vx_1e_4)
    axs[1,1].plot(time, ha1126_vx_0)
    
    axs[1,1].plot(vel_ha1126[:,0], vel_ha1126[:,1] / 100, label='Experimental', color='k')
    
    axs[1,1].set_xlim( (8.5,11) )
    axs[1,1].set_ylim( (-1.5,1.1) )

    axs[1,1].set_title('ADCP HA1126', fontsize='medium')
    axs[1,1].set_ylabel('$u$ (m/s)')
    
    main_labels = [
        'GPU-MWDG2, $\epsilon = 10^{-3}$',
        'GPU-MWDG2, $\epsilon = 10^{-4}$',
        'GPU-DG2',
        'Experimental'
    ]
    
    axs[0,0].legend(
        handles=lines,
        labels=main_labels,
        bbox_to_anchor=(1.7,1.55),
        ncol=2,
        fontsize='x-small'
    )
    
    fig.savefig('predictions.svg', bbox_inches='tight')
    
A4 = (8.3, 11.7)

def plot_speedups():
    print('Plotting speedups...')
    
    cumu_files = [
        os.path.join('eps-1e-3', 'res.cumu'),
        os.path.join('eps-1e-4', 'res.cumu'),
        os.path.join('eps-0',    'res.cumu')
    ]
    
    fig, axs = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=(A4[0]-2, A4[1]-6),
        sharex=True
    )
    
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    ax_wet_cells = axs[0,0]
    ax_rel_dg2   = axs[1,0]
    ax_inst_dg2  = axs[2,0]
    ax_inst_mra  = axs[3,0]
    ax_dt        = axs[4,0]
    ax_num_tstep = axs[0,1]
    ax_cumu_dg2  = axs[1,1]
    ax_cumu_mra  = axs[2,1]
    ax_total     = axs[3,1]
    ax_speedup   = axs[4,1]
    
    ax_wet_cells.sharey(ax_rel_dg2)
    
    ax_wet_cells.set_ylabel('A (%)')
    ax_rel_dg2.set_ylabel('$R_{DG2}$ (%)')
    ax_inst_dg2.set_ylabel('$I_{DG2}$ (ms)')
    ax_inst_mra.set_ylabel('$I_{MRA}$ (ms)')
    ax_dt.set_ylabel('$\Delta t$ (s)')
    ax_num_tstep.set_ylabel('$N_{\Delta t}$ (-)')
    ax_cumu_dg2.set_ylabel('$C_{DG2}$ (s)')
    ax_cumu_mra.set_ylabel('$C_{MRA}$ (s)')
    ax_total.set_ylabel('$C_{tot}$ (s)')
    ax_speedup.set_ylabel('Speedup')
    
    num_yticks = 5
    num_xticks = 10
    
    for ax in axs[-1]:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(num_xticks))
        ax.set_xlabel('$t$ (hr)')
        ax.tick_params(axis='x', labelsize='x-small')
    
    for ax in axs.flat:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(num_yticks))
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4,3))
        ax.yaxis.get_offset_text().set_fontsize('x-small')
        ax.tick_params(labelsize='x-small')
        ax.grid(True)
        
    linewidth = 1
    lw = linewidth
    
    unif_cumu_df = pd.read_csv(cumu_files[-1])[1:]
    
    num_cells_finest = 701 * 692
    
    for cumu_file in cumu_files:
        if cumu_file == os.path.join('eps-0', 'res.cumu'):
            continue
        
        cumu_df = pd.read_csv(cumu_file)[1:]
        
        time_hrs = cumu_df['simtime'] / 3600 + 7
        active_wet = 100 * cumu_df['num_wet_cells'] / unif_cumu_df['num_wet_cells']
        rel_dg2 = 100 * cumu_df['inst_time_solver'] / unif_cumu_df['inst_time_solver']
        speedup = unif_cumu_df['cumu_time_solver'] / cumu_df['runtime_total']
        
        ax_wet_cells.plot(time_hrs, active_wet,                  linewidth=lw)
        ax_rel_dg2.plot  (time_hrs, rel_dg2,                     linewidth=lw)
        ax_inst_dg2.plot (time_hrs, 1000 * cumu_df['inst_time_solver'], linewidth=lw)
        ax_inst_mra.plot (time_hrs, 1000 * cumu_df['inst_time_mra'],    linewidth=lw)
        ax_dt.plot       (time_hrs, cumu_df['dt'],               linewidth=lw)
        ax_num_tstep.plot(time_hrs, cumu_df['num_timesteps'],    linewidth=lw)
        ax_cumu_mra.plot (time_hrs, cumu_df['cumu_time_mra'],    linewidth=lw)
        ax_cumu_dg2.plot (time_hrs, cumu_df['cumu_time_solver'], linewidth=lw)
        ax_total.plot    (time_hrs, cumu_df['runtime_total'],    linewidth=lw)
        ax_speedup.plot  (time_hrs, speedup,                     linewidth=lw)
    
    ax_inst_dg2.plot (time_hrs, 1000 * unif_cumu_df['inst_time_solver'], linewidth=lw)
    ax_dt.plot       (time_hrs, unif_cumu_df['dt'],               linewidth=lw)
    ax_num_tstep.plot(time_hrs, unif_cumu_df['num_timesteps'],    linewidth=lw)
    ax_total.plot    (time_hrs, unif_cumu_df['cumu_time_solver'], linewidth=lw)
    fig.savefig('speedups-hilo.svg', bbox_inches='tight')
    fig.savefig('speedups-hilo.png', bbox_inches='tight')
    plt.show()

def main():
    plot_speedups()
    #plot_predictions()
    
if __name__ == '__main__':
    main()
