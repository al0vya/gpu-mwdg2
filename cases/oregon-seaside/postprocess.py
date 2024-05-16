import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def load_experimental_timeseries(gauges):
    exp_data = collections.defaultdict(dict)
    
    wavegage = np.loadtxt(fname=os.path.join('comparison-data', 'Wavegage.txt'), skiprows=1)
    
    location_data = {
        gauge : np.loadtxt(fname=os.path.join('comparison-data', 'Location_' + gauge + '.txt'), skiprows=3)
        for gauge in gauges[4:] # skip W gauges
    }
    
    for i, gauge in enumerate(gauges[:4]):
        exp_data[gauge]['t'] = wavegage[:,0]
        exp_data[gauge]['e'] = wavegage[:,3+i]
        exp_data[gauge]['u'] = wavegage[:,3+i]
        exp_data[gauge]['M'] = wavegage[:,3+i]
     
    for gauge in gauges[4:]:
        exp_data[gauge]['t'] = location_data[gauge][:,0]
        exp_data[gauge]['e'] = location_data[gauge][:,1]
        exp_data[gauge]['u'] = location_data[gauge][:,2]
        exp_data[gauge]['M'] = location_data[gauge][:,3]
    
    return exp_data

def load_computed_stage_timeseries(
    dirroot,
    gauges
):
    print('Loading computed stage timeseries: %s...' % dirroot)
    
    gauge_data = np.loadtxt(os.path.join(dirroot, 'res.stage'), skiprows=42, delimiter=' ')
    
    return { key : gauge_data[:,i] for i, key in enumerate(['t', 'BD'] + gauges) }
    
def load_computed_velocity_timeseries(
    dirroot,
    gauges
):
    print('Loading computed velocity timeseries: %s...' % dirroot)
    
    gauge_data = np.loadtxt(os.path.join(dirroot, 'res.xvelocity'), skiprows=42, delimiter=' ')
    
    return { key : gauge_data[:,i] for i, key in enumerate(['t', 'BD'] + gauges) }
    
def load_all_computed_timeseries(
    dirroots,
    epsilons,
    gauges
):
    all_computed_stage_timeseries = {
        epsilon : load_computed_stage_timeseries(dirroot, gauges) for epsilon, dirroot in zip(epsilons, dirroots)
    }
    
    all_computed_velocity_timeseries = {
        epsilon : load_computed_velocity_timeseries(dirroot, gauges) for epsilon, dirroot in zip(epsilons, dirroots)
    }
    
    # initialising nested dicts
    all_computed_momentum_timeseries = {
        epsilon : {
            gauge :
                  all_computed_stage_timeseries[epsilon][gauge]
                * all_computed_velocity_timeseries[epsilon][gauge]
                * all_computed_velocity_timeseries[epsilon][gauge]
            for gauge in gauges
        }
        for epsilon in epsilons
    }
    
    for epsilon in epsilons:
        all_computed_momentum_timeseries[epsilon]['t'] = all_computed_stage_timeseries[epsilon]['t']
    
    return {
        'e' : all_computed_stage_timeseries,
        'u' : all_computed_velocity_timeseries,
        'M' : all_computed_momentum_timeseries
    }

def read_stage_elevations(
    dirroot,
    gauges
):
    header = []
    
    with open(os.path.join(dirroot, 'res.stage'), 'r') as fp:
        for i, line in enumerate(fp):
            if i > 2:
                header.append(line)
                
            if i > 38:
                break
    
    return { gauge : float( header[i].split()[3] ) for i, gauge in enumerate(['BD'] + gauges) }
    
def load_all_computed_maps(
    dirroots,
    epsilons
):
    print('Loading computed raster maps...')
    
    all_wd_maps = {
        epsilon : np.loadtxt(fname=os.path.join(dirroot, 'res-1.wd'), skiprows=6) for dirroot, epsilon in zip(dirroots, epsilons)
    }
    
    all_vx_maps = {
        epsilon : np.loadtxt(fname=os.path.join(dirroot, 'res-1.vx'), skiprows=6) for dirroot, epsilon in zip(dirroots, epsilons)
    }
    
    all_Mx_maps = {epsilon : np.multiply( all_wd_maps[epsilon], np.square( all_vx_maps[epsilon] ) ) for epsilon in epsilons}
    
    return {
        'e' : all_wd_maps,
        'u' : all_vx_maps,
        'M' : all_Mx_maps
    }

def compute_RMSE_timeseries(
    all_computed_timeseries,
    gauge
):
    RMSE = collections.defaultdict(dict)
    
    RMSE['e'][1e-3] = np.sqrt( np.square( all_computed_timeseries['e'][1e-3][gauge] - all_computed_timeseries['e'][0][gauge] ).mean() )
    RMSE['e'][1e-4] = np.sqrt( np.square( all_computed_timeseries['e'][1e-4][gauge] - all_computed_timeseries['e'][0][gauge] ).mean() )
    RMSE['u'][1e-3] = np.sqrt( np.square( all_computed_timeseries['u'][1e-3][gauge] - all_computed_timeseries['u'][0][gauge] ).mean() )
    RMSE['u'][1e-4] = np.sqrt( np.square( all_computed_timeseries['u'][1e-4][gauge] - all_computed_timeseries['u'][0][gauge] ).mean() )
    RMSE['M'][1e-3] = np.sqrt( np.square( all_computed_timeseries['M'][1e-3][gauge] - all_computed_timeseries['M'][0][gauge] ).mean() )
    RMSE['M'][1e-4] = np.sqrt( np.square( all_computed_timeseries['M'][1e-4][gauge] - all_computed_timeseries['M'][0][gauge] ).mean() )
    
    return pd.DataFrame(
        data=[[RMSE['e'][1e-3], RMSE['e'][1e-4]], [RMSE['u'][1e-3], RMSE['u'][1e-4]], [RMSE['M'][1e-3], RMSE['M'][1e-4]]],
        index=['h + z', 'vx', 'Mx'],
        columns=['\epsilon = 10-3', '\epsilon = 10-4']
    )

def compute_corr_timeseries(
    all_computed_timeseries,
    gauge
):
    corr = collections.defaultdict(dict)
    
    corr['e'][1e-3] = np.corrcoef( x=all_computed_timeseries['e'][1e-3][gauge], y=all_computed_timeseries['e'][0][gauge] )[0][1]
    corr['e'][1e-4] = np.corrcoef( x=all_computed_timeseries['e'][1e-4][gauge], y=all_computed_timeseries['e'][0][gauge] )[0][1]
    corr['u'][1e-3] = np.corrcoef( x=all_computed_timeseries['u'][1e-3][gauge], y=all_computed_timeseries['u'][0][gauge] )[0][1]
    corr['u'][1e-4] = np.corrcoef( x=all_computed_timeseries['u'][1e-4][gauge], y=all_computed_timeseries['u'][0][gauge] )[0][1]
    corr['M'][1e-3] = np.corrcoef( x=all_computed_timeseries['M'][1e-3][gauge], y=all_computed_timeseries['M'][0][gauge] )[0][1]
    corr['M'][1e-4] = np.corrcoef( x=all_computed_timeseries['M'][1e-4][gauge], y=all_computed_timeseries['M'][0][gauge] )[0][1]
    
    return pd.DataFrame(
        data=[[corr['e'][1e-3], corr['e'][1e-4]], [corr['u'][1e-3], corr['u'][1e-4]], [corr['M'][1e-3], corr['M'][1e-4]]],
        index=['h + z', 'vx', 'Mx'],
        columns=['\epsilon = 10-3', '\epsilon = 10-4']
    )

def compute_RMSE_maps(
    all_computed_maps
):
    RMSE = collections.defaultdict(dict)
    
    RMSE['e'][1e-3] = np.sqrt( np.square( all_computed_maps['e'][1e-3] - all_computed_maps['e'][0] ).mean() )
    RMSE['e'][1e-4] = np.sqrt( np.square( all_computed_maps['e'][1e-4] - all_computed_maps['e'][0] ).mean() )
    RMSE['u'][1e-3] = np.sqrt( np.square( all_computed_maps['u'][1e-3] - all_computed_maps['u'][0] ).mean() )
    RMSE['u'][1e-4] = np.sqrt( np.square( all_computed_maps['u'][1e-4] - all_computed_maps['u'][0] ).mean() )
    RMSE['M'][1e-3] = np.sqrt( np.square( all_computed_maps['M'][1e-3] - all_computed_maps['M'][0] ).mean() )
    RMSE['M'][1e-4] = np.sqrt( np.square( all_computed_maps['M'][1e-4] - all_computed_maps['M'][0] ).mean() )
    
    return pd.DataFrame(
        data=[[RMSE['e'][1e-3], RMSE['e'][1e-4]], [RMSE['u'][1e-3], RMSE['u'][1e-4]], [RMSE['M'][1e-3], RMSE['M'][1e-4]]],
        index=['h + z', 'vx', 'Mx'],
        columns=['\epsilon = 10-3', '\epsilon = 10-4']
    )
    
def compute_corr_maps(
    all_computed_maps
):
    corr = collections.defaultdict(dict)
    
    corr['e'][1e-3] = np.corrcoef( x=all_computed_maps['e'][1e-3].flatten(), y=all_computed_maps['e'][0].flatten() )[0][1]
    corr['e'][1e-4] = np.corrcoef( x=all_computed_maps['e'][1e-4].flatten(), y=all_computed_maps['e'][0].flatten() )[0][1]
    corr['u'][1e-3] = np.corrcoef( x=all_computed_maps['u'][1e-3].flatten(), y=all_computed_maps['u'][0].flatten() )[0][1]
    corr['u'][1e-4] = np.corrcoef( x=all_computed_maps['u'][1e-4].flatten(), y=all_computed_maps['u'][0].flatten() )[0][1]
    corr['M'][1e-3] = np.corrcoef( x=all_computed_maps['M'][1e-3].flatten(), y=all_computed_maps['M'][0].flatten() )[0][1]
    corr['M'][1e-4] = np.corrcoef( x=all_computed_maps['M'][1e-4].flatten(), y=all_computed_maps['M'][0].flatten() )[0][1]
    
    return pd.DataFrame(
        data=[[corr['e'][1e-3], corr['e'][1e-4]], [corr['u'][1e-3], corr['u'][1e-4]], [corr['M'][1e-3], corr['M'][1e-4]]],
        index=['h + z', 'vx', 'Mx'],
        columns=['\epsilon = 10-3', '\epsilon = 10-4']
    )

def write_table(
    all_computed_timeseries,
    all_computed_maps
):
    print('Writing table...')
    
    corr_A1   = compute_corr_timeseries(all_computed_timeseries, gauge='A1')
    corr_B6   = compute_corr_timeseries(all_computed_timeseries, gauge='B6')
    corr_D4   = compute_corr_timeseries(all_computed_timeseries, gauge='D4')
    corr_maps = compute_corr_maps(all_computed_maps)
    
    RMSE_A1   = compute_RMSE_timeseries(all_computed_timeseries, gauge='A1')
    RMSE_B6   = compute_RMSE_timeseries(all_computed_timeseries, gauge='B6')
    RMSE_D4   = compute_RMSE_timeseries(all_computed_timeseries, gauge='D4')
    RMSE_maps = compute_RMSE_maps(all_computed_maps)
    
    RMSE_corr_A1   = pd.concat([RMSE_A1,   corr_A1],   axis=1, keys=['RMSE', 'R2'])
    RMSE_corr_B6   = pd.concat([RMSE_B6,   corr_B6],   axis=1, keys=['RMSE', 'R2'])
    RMSE_corr_D4   = pd.concat([RMSE_D4,   corr_D4],   axis=1, keys=['RMSE', 'R2'])
    RMSE_corr_maps = pd.concat([RMSE_maps, corr_maps], axis=1, keys=['RMSE', 'R2'])
    
    RMSE_corr = pd.concat([RMSE_corr_A1, RMSE_corr_B6, RMSE_corr_D4, RMSE_corr_maps], keys=['Time series at A1', 'Time series at B6', 'Time series at D4', 'Spatial map at ts'])
    
    RMSE_corr.to_csv('table.csv')
    
def plot_timeseries_at_gauge(
    gauge,
    all_computed_timeseries,
    exp_data,
    epsilons,
    outer_subplot_spec,
    include_legend=False
):
    print('Plotting timeseries at gauge: ' + gauge)
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # to find a way to iterate over epsilons and epsilons_vx
    iterator_stage = zip(
        all_computed_timeseries['e'].keys(),
        all_computed_timeseries['u'].keys(),
        colors
    )
    
    T = 0
    
    axs = outer_subplot_spec.subgridspec(nrows=3, ncols=1, hspace=0.3).subplots()
    
    lines = []
    
    for epsilon, color in zip(epsilons, colors):
        # computed timeseries
        line, = axs[0].plot(
            all_computed_timeseries['e'][epsilon]['t'] + T,
            all_computed_timeseries['e'][epsilon][gauge],
            linewidth=1
        )
        
        lines.append(line)
        
        axs[1].plot(
            all_computed_timeseries['u'][epsilon]['t'] + T,
            all_computed_timeseries['u'][epsilon][gauge],
            linewidth=1
        )
        
        axs[2].plot(
            all_computed_timeseries['M'][epsilon]['t'] + T,
            all_computed_timeseries['M'][epsilon][gauge],
            linewidth=1
        )
        
    # experimental data
    line, = axs[0].plot(
        exp_data[gauge]['t'],
        exp_data[gauge]['e'],
        linewidth=0.75,
        color='k'
    )
    
    axs[1].plot(
        exp_data[gauge]['t'],
        exp_data[gauge]['u'],
        linewidth=0.75,
        color='k'
    )
    
    axs[2].plot(
        exp_data[gauge]['t'][:-1000],
        exp_data[gauge]['M'][:-1000], # to avoid spike
        linewidth=0.75,
        color='k'
    )
    
    lines.append(line)
    
    if include_legend:
        main_labels = [
            'GPU-MWDG2, $\epsilon = 10^{-3}$',
            'GPU-MWDG2, $\epsilon = 10^{-4}$',
            'GPU-DG2',
            'Experimental'
        ]
        
        axs[0].legend(
            handles=lines,
            labels=main_labels,
            bbox_to_anchor=(3.8, 2.2),
            ncol=2
        )
    
    plt.setp(
        axs[0],
        xlim=(20,40),
        xlabel='$t$ (s)',
        ylabel='$h + z$ (m)'
    )
    
    plt.setp(
        axs[1],
        xlim=(20,40),
        xlabel='$t$ (s)',
        ylabel='$u$ (ms$^{-1}$)'
    )
        
    plt.setp(
        axs[2],
        xlim=(20,40),
        xlabel='$t$ (s)',
        ylabel='$M_x$ (m$^3$s$^{-2}$)'
    )
   
def plot_main_gauge_timeseries(
    exp_data,
    epsilons,
    all_computed_timeseries
):
    print('Plotting main gauge data...')
    
    fig = plt.figure(
        figsize=(6,4)
    )
    
    outer_gridspec = fig.add_gridspec(
        nrows=1, 
        ncols=3,
        wspace=0.7
    )
    
    axs = outer_gridspec.subplots()
    
    # plotting the gauge data
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    
    axs[0].set_title('Point A1')
    axs[1].set_title('Point B6')
    axs[2].set_title('Point D4')
    
    plot_timeseries_at_gauge(
        gauge='A1',
        all_computed_timeseries=all_computed_timeseries,
        exp_data=exp_data,
        epsilons=epsilons,
        outer_subplot_spec=outer_gridspec[0],
        include_legend=True
    )
    
    plot_timeseries_at_gauge(
        gauge='B6',
        all_computed_timeseries=all_computed_timeseries,
        exp_data=exp_data,
        epsilons=epsilons,
        outer_subplot_spec=outer_gridspec[1]
    )
    
    plot_timeseries_at_gauge(
        gauge='D4',
        all_computed_timeseries=all_computed_timeseries,
        exp_data=exp_data,
        epsilons=epsilons,
        outer_subplot_spec=outer_gridspec[2]
    )
    
    fig.savefig('predictions.svg', bbox_inches='tight')

A4 = (8.3, 11.7)

def plot_speedups(
    dirroots,
    epsilons
):
    print('Plotting speedups...')
    
    fig, axs = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(A4[0]-2, A4[1]-6),
        sharex=True
    )
      
    ax_wet_cells    = axs[0,0]; 
    ax_rel_dg2      = axs[0,1]; ax_rel_dg2.sharey(ax_wet_cells)
    ax_rel_mra      = axs[0,2]; ax_rel_mra.sharey(ax_wet_cells)
    ax_inst_speedup = axs[1,0]
    ax_dt           = axs[1,1]
    ax_num_tstep    = axs[1,2]
    ax_cumu_split   = axs[2,0]
    ax_total        = axs[2,1]
    ax_speedup      = axs[2,2]
    
    ax_wet_cells.set_title('$N_{wet}$ (%)',                      fontsize='small')
    ax_rel_dg2.set_title('$R_{DG2}$ (%)',                        fontsize='small')
    ax_rel_mra.set_title('$R_{MRA}$ (%)',                        fontsize='small')
    ax_inst_speedup.set_title('$S_{inst}$ (-)',                  fontsize='small')
    ax_dt.set_title('$\Delta t$ (s)',                            fontsize='small')
    ax_num_tstep.set_title('$N_{\Delta t}$ (-)',                 fontsize='small')
    ax_cumu_split.set_title('$C_{MRA}$ (dotted), $C_{DG2}$ (s)', fontsize='small')
    ax_total.set_title('$C_{tot}$ (s)',                          fontsize='small')
    ax_speedup.set_title('$S_{acc}$ (-)',                        fontsize='small')
    
    linewidth = 1
    lw = linewidth
    
    unif_cumu_df = pd.read_csv(os.path.join('eps-0', 'res.cumu'))[1:]
    
    num_cells_unif = 2181 * 1091
    
    for dirroot, epsilon in zip(dirroots, epsilons):
        if epsilon == 0:
            continue
        
        cumu_df = pd.read_csv(os.path.join(dirroot, 'res.cumu'))[1:]
        
        wet_cells    = cumu_df['num_wet_cells']    / unif_cumu_df['num_wet_cells']
        rel_dg2      = cumu_df['inst_time_solver'] / unif_cumu_df['inst_time_solver']
        rel_mra      = cumu_df['inst_time_mra']    / unif_cumu_df['inst_time_solver']
        inst_speedup = 1 / (rel_dg2 + rel_mra)
        speedup      = unif_cumu_df['cumu_time_solver'] / cumu_df['runtime_total']
        
        ax_wet_cells.plot   (cumu_df['simtime'], 100 * wet_cells,             linewidth=lw)
        ax_rel_dg2.plot     (cumu_df['simtime'], 100 * rel_dg2,               linewidth=lw)
        ax_rel_mra.plot     (cumu_df['simtime'], 100 * rel_mra,               linewidth=lw)
        ax_inst_speedup.plot(cumu_df['simtime'], inst_speedup,                linewidth=lw)
        ax_dt.plot          (cumu_df['simtime'], cumu_df['dt'],               linewidth=lw)
        ax_num_tstep.plot   (cumu_df['simtime'], cumu_df['num_timesteps'],    linewidth=lw)
        ax_cumu_split.plot  (cumu_df['simtime'], cumu_df['cumu_time_solver'], linewidth=lw)
        ax_cumu_split.scatter(cumu_df['simtime'].iloc[::8], cumu_df['cumu_time_mra'].iloc[::8], marker='x', s=4, color=ax_cumu_split.get_lines()[-1].get_color())
        ax_total.plot       (cumu_df['simtime'], cumu_df['runtime_total'],    linewidth=lw)
        ax_speedup.plot     (cumu_df['simtime'], speedup,                     linewidth=lw)
        
    ax_dt.plot       (unif_cumu_df['simtime'], unif_cumu_df['dt'],               linewidth=lw)
    ax_num_tstep.plot(unif_cumu_df['simtime'], unif_cumu_df['num_timesteps'],    linewidth=lw)
    ax_total.plot    (unif_cumu_df['simtime'], unif_cumu_df['cumu_time_solver'], linewidth=lw)
    
    num_yticks = 5
    num_xticks = 5
    
    for ax in axs[-1]:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(num_xticks))
        ax.set_xlabel('$t$ (s)')
        ax.tick_params(axis='x', labelsize='small')
    
    for ax in axs.flat:
        ax.set_xlim((0, 40))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(num_yticks))
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,3), useMathText=True)
        ax.yaxis.get_offset_text().set_fontsize('small')
        ax.tick_params(labelsize='small')
        ax.grid(True)
        
    ax_wet_cells.legend(handles=ax_dt.get_lines(), labels=["$\epsilon = 10^{-3}$", "$\epsilon = 10^{-4}$", "GPU-DG2"], fontsize='x-small', loc='upper left')
    
    fig.tight_layout()
    fig.savefig('speedups-oregon-seaside.png', bbox_inches='tight')
    fig.savefig('speedups-oregon-seaside.svg', bbox_inches='tight')
    
def main():
    epsilons = [1e-3, 1e-4, 0]
    
    dirroots = [
        'eps-1e-3',
        'eps-1e-4',
        'eps-0'
    ]
    
    gauges = [
        'W1', 'W2', 'W3', 'W4',
        'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',
        'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',
        'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
        'D1', 'D2', 'D3', 'D4'
    ]
    
    #exp_data                = load_experimental_timeseries(gauges)
    #all_computed_timeseries = load_all_computed_timeseries(dirroots, epsilons, gauges)
    #all_computed_maps       = load_all_computed_maps(dirroots, epsilons)
    
    #write_table(all_computed_timeseries, all_computed_maps)
    
    plot_speedups(dirroots, epsilons)
    
    #plot_main_gauge_timeseries(exp_data, epsilons, all_computed_timeseries)
    
if __name__ == '__main__':
    main()
