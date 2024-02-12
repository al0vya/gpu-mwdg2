import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_experimental_timeseries(gauges):
    exp_data = collections.defaultdict(dict)
    
    wavegage = np.loadtxt(fname=os.path.join("comparison-data", "Wavegage.txt"), skiprows=1)
    
    location_data = {
        gauge : np.loadtxt(fname=os.path.join("comparison-data", "Location_" + gauge + ".txt"), skiprows=3)
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
    print("Loading computed stage timeseries: %s..." % dirroot)
    
    gauge_data = np.loadtxt(os.path.join(dirroot, "res.stage"), skiprows=42, delimiter=" ")
    
    return { key : gauge_data[:,i] for i, key in enumerate(['t', "BD"] + gauges) }
    
def load_computed_velocity_timeseries(
    dirroot,
    gauges
):
    print("Loading computed velocity timeseries: %s..." % dirroot)
    
    gauge_data = np.loadtxt(os.path.join(dirroot, "res.xvelocity"), skiprows=42, delimiter=" ")
    
    return { key : gauge_data[:,i] for i, key in enumerate(['t', "BD"] + gauges) }
    
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
    
    with open(os.path.join(dirroot, "res.stage"), 'r') as fp:
        for i, line in enumerate(fp):
            if i > 2:
                header.append(line)
                
            if i > 38:
                break
    
    return { gauge : float( header[i].split()[3] ) for i, gauge in enumerate(["BD"] + gauges) }
    
def load_all_computed_maps(
    dirroots,
    epsilons
):
    print("Loading computed raster maps...")
    
    all_wd_maps = {
        epsilon : np.loadtxt(fname=os.path.join(dirroot, "res-1.wd"), skiprows=6) for dirroot, epsilon in zip(dirroots, epsilons)
    }
    
    all_vx_maps = {
        epsilon : np.loadtxt(fname=os.path.join(dirroot, "res-1.vx"), skiprows=6) for dirroot, epsilon in zip(dirroots, epsilons)
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
        index=["h + z", "vx", "Mx"],
        columns=["\epsilon = 10-3", "\epsilon = 10-4"]
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
        index=["h + z", "vx", "Mx"],
        columns=["\epsilon = 10-3", "\epsilon = 10-4"]
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
        index=["h + z", "vx", "Mx"],
        columns=["\epsilon = 10-3", "\epsilon = 10-4"]
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
        index=["h + z", "vx", "Mx"],
        columns=["\epsilon = 10-3", "\epsilon = 10-4"]
    )

def write_table(
    all_computed_timeseries,
    all_computed_maps
):
    print("Writing table...")
    
    corr_A1   = compute_corr_timeseries(all_computed_timeseries, gauge="A1")
    corr_B6   = compute_corr_timeseries(all_computed_timeseries, gauge="B6")
    corr_D4   = compute_corr_timeseries(all_computed_timeseries, gauge="D4")
    corr_maps = compute_corr_maps(all_computed_maps)
    
    RMSE_A1   = compute_RMSE_timeseries(all_computed_timeseries, gauge="A1")
    RMSE_B6   = compute_RMSE_timeseries(all_computed_timeseries, gauge="B6")
    RMSE_D4   = compute_RMSE_timeseries(all_computed_timeseries, gauge="D4")
    RMSE_maps = compute_RMSE_maps(all_computed_maps)
    
    RMSE_corr_A1   = pd.concat([RMSE_A1,   corr_A1],   axis=1, keys=["RMSE", "R2"])
    RMSE_corr_B6   = pd.concat([RMSE_B6,   corr_B6],   axis=1, keys=["RMSE", "R2"])
    RMSE_corr_D4   = pd.concat([RMSE_D4,   corr_D4],   axis=1, keys=["RMSE", "R2"])
    RMSE_corr_maps = pd.concat([RMSE_maps, corr_maps], axis=1, keys=["RMSE", "R2"])
    
    RMSE_corr = pd.concat([RMSE_corr_A1, RMSE_corr_B6, RMSE_corr_D4, RMSE_corr_maps], keys=["Time series at A1", "Time series at B6", "Time series at D4", "Spatial map at ts"])
    
    RMSE_corr.to_csv("table.csv")
    
def plot_timeseries_at_gauge(
    gauge,
    all_computed_timeseries,
    exp_data,
    epsilons,
    outer_subplot_spec,
    include_legend=False
):
    print("Plotting timeseries at gauge: " + gauge)
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
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
            "GPU-MWDG2, $\epsilon = 10^{-3}$",
            "GPU-MWDG2, $\epsilon = 10^{-4}$",
            "GPU-DG2",
            "Experimental"
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
        xlabel="$t$ (s)",
        ylabel="$h + z$ (m)"
    )
    
    plt.setp(
        axs[1],
        xlim=(20,40),
        xlabel="$t$ (s)",
        ylabel="$u$ (ms$^{-1}$)"
    )
        
    plt.setp(
        axs[2],
        xlim=(20,40),
        xlabel="$t$ (s)",
        ylabel="$M_x$ (m$^3$s$^{-2}$)"
    )
   
def plot_main_gauge_timeseries(
    exp_data,
    epsilons,
    all_computed_timeseries
):
    print("Plotting main gauge data...")
    
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
    axs[0].axis("off")
    axs[1].axis("off")
    axs[2].axis("off")
    
    axs[0].set_title("Point A1")
    axs[1].set_title("Point B6")
    axs[2].set_title("Point D4")
    
    plot_timeseries_at_gauge(
        gauge="A1",
        all_computed_timeseries=all_computed_timeseries,
        exp_data=exp_data,
        epsilons=epsilons,
        outer_subplot_spec=outer_gridspec[0],
        include_legend=True
    )
    
    plot_timeseries_at_gauge(
        gauge="B6",
        all_computed_timeseries=all_computed_timeseries,
        exp_data=exp_data,
        epsilons=epsilons,
        outer_subplot_spec=outer_gridspec[1]
    )
    
    plot_timeseries_at_gauge(
        gauge="D4",
        all_computed_timeseries=all_computed_timeseries,
        exp_data=exp_data,
        epsilons=epsilons,
        outer_subplot_spec=outer_gridspec[2]
    )
    
    fig.savefig("predictions", bbox_inches="tight")

class YAxisLimits:
    def __init__(self):
        limits = {}
        
        limits["reduction"]   = {}
        limits["frac_DG2"]    = {}
        limits["rel_speedup"] = {}
        
        for key in limits:
            limits[key]["min"] =  (2 ** 63 + 1)
            limits[key]["max"] = -(2 ** 63 + 1)
        
        limits["reduction"]["min"] = 0
        
        self.limits = limits
    
    def set_y_axis_limits(
        self,
        field,
        field_data
    ):
        self.limits[field]["min"] = min(
            self.limits[field]["min"],
            min(field_data)
        )
        
        self.limits[field]["max"] = max(
            self.limits[field]["max"],
            max(field_data)
        )
    
    def get_y_axis_ticks(
        self,
        field,
        num_ticks
    ):
        if field == "rel_speedup":
            min_val = 0.9 * self.limits[field]["min"]
            max_val = 1.1 * self.limits[field]["max"]
        else:
            min_val = max( 0,   0.9 * self.limits[field]["min"] )
            max_val = min( 100, 1.1 * self.limits[field]["max"] )
        
        d_val = (max_val - min_val) / num_ticks
        
        return [ min_val + i * d_val for i in range(num_ticks+1) ]
        
    def set_y_axis_ticks(
        self,
        ax,
        field,
        num_ticks,
        num_digits_round=1
    ):
        yticks = self.get_y_axis_ticks(
            field=field,
            num_ticks=num_ticks
        )
        
        ax.set_yticks( [] )
        ax.set_yticks(
            ticks=yticks,
            minor=False
        )
        
        yticks = [round(ytick, num_digits_round) if field == "rel_speedup" else int(ytick) for ytick in yticks]
        
        ax.set_yticklabels(
            labels=yticks,
            minor=False
        )

def plot_speedups(
    dirroots,
    epsilons
):
    print("Plotting speedups...")
    
    ref_runtime = np.loadtxt(
        fname=os.path.join("eps-0", "res.cumu"),
        skiprows=1,
        delimiter=','
    )[:,3]
    
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(6, 4),
        sharex=True
    )
    
    gridspec = axs[0,0].get_gridspec()
    
    axs[0,0].remove()
    axs[1,0].remove()
    
    ax_reduction   = fig.add_subplot( gridspec[:,0] )
    ax_frac_DG2    = axs[0,1]
    ax_rel_speedup = axs[1,1]
    
    axs = [ax_reduction, ax_frac_DG2, ax_rel_speedup]
    
    y_axis_limits = YAxisLimits()
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    lines = []
    
    for epsilon, dirroot, color in zip(epsilons, dirroots, colors):
        if epsilon == 0:
            continue
        
        cumulative_data = np.loadtxt(fname=os.path.join(dirroot, "res.cumu"), skiprows=1, delimiter=',')
        
        time = cumulative_data[:,0]
        
        rel_speedup = ref_runtime / cumulative_data[:,3]
        
        compression = 100 * cumulative_data[:,5]
        
        ax_rel_speedup.plot(
            time,
            rel_speedup,
            linewidth=2
        )
        
        y_axis_limits.set_y_axis_limits(field="rel_speedup", field_data=rel_speedup)
        
        ax_rel_speedup.set_ylabel("$S_{rel}$ (-)")
        
        line, = ax_reduction.plot(
            time,
            compression,
            linewidth=2,
            color=color
        )
        
        ax_reduction.plot(
            [ time[0], round(time[-1], 0) ],
            [ compression[0], compression[0] ],
            linestyle='--',
            linewidth=1.5,
            color=color
        )
        
        y_axis_limits.set_y_axis_limits(field="reduction", field_data=compression)
        
        ax_reduction.set_ylabel("$R_{cell}$ (%)")
        
        frac_DG2 = 100 * (
            cumulative_data[:,2]
            /
            cumulative_data[:,3]
        )
        
        ax_frac_DG2.plot(
            time[1:],
            frac_DG2[1:],
            linewidth=2
        )
        
        y_axis_limits.set_y_axis_limits(field="frac_DG2", field_data=frac_DG2[1:])
        
        ax_frac_DG2.set_ylabel("$F_{DG2}$ (%)")
        
        lines.append(line)
    
    y_axis_limits.set_y_axis_ticks(ax=ax_rel_speedup, field="rel_speedup", num_ticks=5, num_digits_round=1)
    y_axis_limits.set_y_axis_ticks(ax=ax_reduction,   field="reduction",   num_ticks=10)
    y_axis_limits.set_y_axis_ticks(ax=ax_frac_DG2,    field="frac_DG2",    num_ticks=5)
    
    xlim = ( 0, round(time[-1], 0) )
    
    for ax in axs:
        ax.set_xlim(xlim)
    
    ax_reduction.set_xlabel("$t$ (s)")
    ax_rel_speedup.set_xlabel("$t$ (s)")
    ax_reduction.legend(handles=lines, labels=["$\epsilon = 10^{-3}$", "$\epsilon = 10^{-4}$"])
    fig.tight_layout()
    fig.savefig(fname="speedups.png", bbox_inches="tight")
    
def main():
    epsilons = [1e-3, 1e-4, 0]
    
    dirroots = [
        "eps-1e-3",
        "eps-1e-4",
        "eps-0"
    ]
    
    gauges = [
        "W1", "W2", "W3", "W4",
        "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9",
        "D1", "D2", "D3", "D4"
    ]
    
    exp_data                = load_experimental_timeseries(gauges)
    all_computed_timeseries = load_all_computed_timeseries(dirroots, epsilons, gauges)
    all_computed_maps       = load_all_computed_maps(dirroots, epsilons)
    
    write_table(all_computed_timeseries, all_computed_maps)
    
    plot_speedups(dirroots, epsilons)
    
    plot_main_gauge_timeseries(exp_data, epsilons, all_computed_timeseries)
    
if __name__ == "__main__":
    main()
