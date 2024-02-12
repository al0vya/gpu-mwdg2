import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

epsilons = [1e-3, 1e-4, 0]

dirroots = [
    "_eps-1e-3",
    "_eps-1e-4",
    "_eps-0"
]

def load_experimental_stage_timeseries():
    print("Loading experimental stage timeseries...")
    
    port_data  = np.loadtxt( os.path.join("input-data", "port_data.txt" ) )
    tide_gauge = np.loadtxt( os.path.join("input-data", "tide_gauge.txt") )
    
    A_Beacon = {
        "time"    : port_data[:,0+0],
        "tsunami" : port_data[:,1+0],
        "total"   : port_data[:,2+0],
        "tidal"   : port_data[:,2+0] - port_data[:,1+0],
    }
    
    Tug_Harbour = {
        "time"    : port_data[:,0+3],
        "tsunami" : port_data[:,1+3],
        "total"   : port_data[:,2+3],
        "tidal"   : port_data[:,2+3] - port_data[:,1+3],
    }
    
    Sulphur_Point = {
        "time"    : port_data[:,0+6],
        "tsunami" : port_data[:,1+6],
        "total"   : port_data[:,2+6],
        "tidal"   : port_data[:,2+6] - port_data[:,1+6],
    }
    
    Taut = {
        "time"    : tide_gauge[:,0+0],
        "tsunami" : tide_gauge[:,1+0],
        "total"   : tide_gauge[:,2+0],
        "tidal"   : tide_gauge[:,2+0] - tide_gauge[:,1+0],
    }
    
    Moturiki = {
        "time"    : tide_gauge[:,0+3],
        "tsunami" : tide_gauge[:,1+3],
        "total"   : tide_gauge[:,2+3],
        "tidal"   : tide_gauge[:,2+3] - tide_gauge[:,1+3],
    }
    
    return {
        "A Beacon"      : A_Beacon,
        "Tug Harbour"   : Tug_Harbour,
        "Sulphur Point" : Sulphur_Point,
        "Taut"          : Taut,
        "Moturiki"      : Moturiki
    }
    
def load_experimental_velocity_timeseries():
    print("Loading experimental velocity timeseries...")
    
    current_data = np.loadtxt( os.path.join("input-data", "currents.txt") )
    
    return {
        "time" : current_data[:,0],
        "u" : {
            "tsunami" : current_data[:,1],
            "total"   : current_data[:,5],
            "tidal"   : current_data[:,5] - current_data[:,1]
        },
        "v" : {
            "tsunami" : current_data[:,2],
            "total"   : current_data[:,6],
            "tidal"   : current_data[:,6] - current_data[:,2]
        },
        "speed" : {
            "tsunami" : current_data[:,3],
            "total"   : np.sqrt(current_data[:,5]**2 + current_data[:,6]**2),
            "tidal"   : np.sqrt(
              ( current_data[:,5] - current_data[:,1] ) ** 2 # vel_total - vel_tsu = vel_tidal
            + ( current_data[:,6] - current_data[:,2] ) ** 2
            )
        }
    }

def load_computed_stage_timeseries(
    dirroot
):
    print("Loading computed stage timeseries...")
    
    gauges = np.loadtxt(os.path.join(dirroot, "res.stage"), skiprows=11, delimiter=" ")
    
    datum = -37.7
    
    return {
        "time"          : gauges[:,0] / 3600, # get into hours
        "A Beacon"      : gauges[:,1] + datum,
        "Tug Harbour"   : gauges[:,2] + datum,
        "Sulphur Point" : gauges[:,3] + datum,
        "Moturiki"      : gauges[:,4] + datum
    }

def load_computed_maps(
    dirroot
):
    print(f"Loading spatial map at {dirroot}")
    
    el = np.loadtxt(fname=os.path.join(dirroot, "res-1.elev"), skiprows=6)
    vx = np.loadtxt(fname=os.path.join(dirroot, "res-1.vx"),   skiprows=6)
    vy = np.loadtxt(fname=os.path.join(dirroot, "res-1.vy"),   skiprows=6)
    
    speed = np.sqrt(np.square(vx) + np.square(vy))
    
    return {"elev" : el, "speed" : speed}
    
def load_all_computed_maps():
    return { epsilon : load_computed_maps(dirroot) for dirroot, epsilon in zip(dirroots, epsilons) }

def load_all_computed_stage_timeseries():
    return { epsilon : load_computed_stage_timeseries(dirroot) for epsilon, dirroot in zip(epsilons, dirroots) }
    
def load_computed_velocity_timeseries(
    dirroot
):
    print("Loading computed velocity timeseries...")
    
    velocity_timeseries_data_vx = np.loadtxt(os.path.join(dirroot, "res.xvelocity"), skiprows=11, delimiter=" ")
    velocity_timeseries_data_vy = np.loadtxt(os.path.join(dirroot, "res.yvelocity"), skiprows=11, delimiter=" ")
    
    return {
        "time"  : velocity_timeseries_data_vx[:,0] / 3600, # getting into hours
        "speed" : np.sqrt(
              velocity_timeseries_data_vx[:,5] ** 2
            + velocity_timeseries_data_vy[:,5] ** 2
         )
    }

def load_all_computed_velocity_timeseries():
    return { epsilon : load_computed_velocity_timeseries(dirroot) for epsilon, dirroot in zip(epsilons, dirroots) }

def read_stage_point_elevations(
    dirroot
):
    header = []
    
    with open(os.path.join(dirroot, "res.stage"), 'r') as fp:
        for i, line in enumerate(fp):
            if i > 2:
                header.append(line)
                
            if i > 7:
                break
    
    return {
        "A Beacon"      : float( header[0].split()[3] ),
        "Tug Harbour"   : float( header[1].split()[3] ),
        "Sulphur Point" : float( header[2].split()[3] ),
        "Moturiki"      : float( header[3].split()[3] )
    }

def read_all_stage_point_elevations():    
    return { epsilon : read_stage_point_elevations(dirroot) for epsilon, dirroot in zip(epsilons, dirroots) }
    
def compare_stage_timeseries(
    all_computed_stage_timeseries,
    experimental_stage_timeseries,
    all_stage_point_elevations,
    ax,
    name,
    speed=False
):
    if speed:
        compare_velocity_timeseries(ax)
        return
    
    print("Plotting stage timeseries at " + name)
    
    lines = []
    
    for epsilon in all_computed_stage_timeseries:
        stage_point_elevations    = all_stage_point_elevations[epsilon]
        computed_stage_timeseries = all_computed_stage_timeseries[epsilon]
        
        line, = ax.plot(
            computed_stage_timeseries["time"],
            computed_stage_timeseries[name] + stage_point_elevations[name],
            linewidth=0.85
        )
        
        lines.append(line)
    
    line, = ax.plot(
        experimental_stage_timeseries[name]["time"],
        experimental_stage_timeseries[name]["total"],
        linewidth=0.85,
        label="Experimental",
        color='k',
        zorder=3
    )
    
    lines.append(line)
    
    if name == "A Beacon":
        main_labels = [
            "GPU-MWDG2, $\epsilon = 10^{-3}$",
            "GPU-MWDG2, $\epsilon = 10^{-4}$",
            "GPU-DG2",
            "Experimental"
        ]
        
        ax.legend(
            handles=lines,
            labels=main_labels,
            bbox_to_anchor=(0.85, 2.85),
            ncol=2
        )
    
    xmin = 10
    xmax = 35
    
    num_xticks = 10
    
    dx = (xmax - xmin) / num_xticks
    
    xticks = [ round(xmin + dx * i, 1) for i in range(num_xticks+1) ]
    
    ax.set_xticks( [] )
    ax.set_xticks(
        ticks=xticks,
        minor=False
    )
    
    ax.set_xticklabels(
        labels=xticks,
        minor=False
    )
    
    ymin = -1.1
    ymax =  1.1
    
    num_yticks = 4
    
    dy = (ymax - ymin) / num_yticks
    
    yticks = [ round(ymin + dy * i, 1) for i in range(num_yticks+1) ]
    
    ax.set_yticks( [] )
    ax.set_yticks(
        ticks=yticks,
        minor=False
    )
    
    ax.set_yticklabels(
        labels=yticks,
        minor=False
    )
    
    plt.setp(
        ax,
        title=name,
        xlim=(xmin,xmax),
        ylim=(ymin,ymax),
        xlabel="$t$ (hr)",
        ylabel="$h + z$ (m)"
    )
    
def write_table(
    all_computed_stage_timeseries,
    all_computed_maps
):
    names = [
        "A Beacon",
        "Tug Harbour",
        "Sulphur Point",
        "Moturiki"
    ]
    
    # recursive defaultdict from https://stackoverflow.com/questions/19189274/nested-defaultdict-of-defaultdict
    rec_dd = lambda: collections.defaultdict(rec_dd)
    
    errors_e = rec_dd()
    
    for name in names:
        errors_e[1e-3][name]["RMSE"] = np.sqrt( np.square( all_computed_stage_timeseries[1e-3][name] - all_computed_stage_timeseries[0][name] ).mean() )
        errors_e[1e-4][name]["RMSE"] = np.sqrt( np.square( all_computed_stage_timeseries[1e-4][name] - all_computed_stage_timeseries[0][name] ).mean() )
        errors_e[1e-3][name]["corr"] = np.corrcoef( x=all_computed_stage_timeseries[1e-3][name], y=all_computed_stage_timeseries[0][name] )[0][1]
        errors_e[1e-4][name]["corr"] = np.corrcoef( x=all_computed_stage_timeseries[1e-4][name], y=all_computed_stage_timeseries[0][name] )[0][1]
        
    errors_e[1e-3]["elev"]["RMSE"]  = np.sqrt( np.square( all_computed_maps[1e-3]["elev"]  - all_computed_maps[0]["elev"] ).mean() )
    errors_e[1e-4]["elev"]["RMSE"]  = np.sqrt( np.square( all_computed_maps[1e-4]["elev"]  - all_computed_maps[0]["elev"] ).mean() )
    errors_e[1e-3]["speed"]["RMSE"] = np.sqrt( np.square( all_computed_maps[1e-3]["speed"] - all_computed_maps[0]["speed"] ).mean() )
    errors_e[1e-4]["speed"]["RMSE"] = np.sqrt( np.square( all_computed_maps[1e-4]["speed"] - all_computed_maps[0]["speed"] ).mean() )
    
    errors_e[1e-3]["elev"]["corr"]  = np.corrcoef( x=all_computed_maps[1e-3]["elev"].flatten(),  y=all_computed_maps[0]["elev"].flatten() )[0][1]
    errors_e[1e-4]["elev"]["corr"]  = np.corrcoef( x=all_computed_maps[1e-4]["elev"].flatten(),  y=all_computed_maps[0]["elev"].flatten() )[0][1]
    errors_e[1e-3]["speed"]["corr"] = np.corrcoef( x=all_computed_maps[1e-3]["speed"].flatten(), y=all_computed_maps[0]["speed"].flatten() )[0][1]
    errors_e[1e-4]["speed"]["corr"] = np.corrcoef( x=all_computed_maps[1e-4]["speed"].flatten(), y=all_computed_maps[0]["speed"].flatten() )[0][1]
    
    table_data = [
        ( "h + z", errors_e[1e-3]["A Beacon"]["RMSE"],      errors_e[1e-4]["A Beacon"]["RMSE"],      errors_e[1e-3]["A Beacon"]["corr"],      errors_e[1e-4]["A Beacon"]["corr"] ),
        ( "h + z", errors_e[1e-3]["Tug Harbour"]["RMSE"],   errors_e[1e-4]["Tug Harbour"]["RMSE"],   errors_e[1e-3]["Tug Harbour"]["corr"],   errors_e[1e-4]["Tug Harbour"]["corr"] ),
        ( "h + z", errors_e[1e-3]["Sulphur Point"]["RMSE"], errors_e[1e-4]["Sulphur Point"]["RMSE"], errors_e[1e-3]["Sulphur Point"]["corr"], errors_e[1e-4]["Sulphur Point"]["corr"] ),
        ( "h + z", errors_e[1e-3]["Moturiki"]["RMSE"],      errors_e[1e-4]["Moturiki"]["RMSE"],      errors_e[1e-3]["Moturiki"]["corr"],      errors_e[1e-4]["Moturiki"]["corr"] ),
        ( "h + z", errors_e[1e-3]["elev"]["RMSE"],          errors_e[1e-4]["elev"]["RMSE"],          errors_e[1e-3]["elev"]["corr"],          errors_e[1e-4]["elev"]["corr"] ),
        ( "speed",    errors_e[1e-3]["speed"]["RMSE"],      errors_e[1e-4]["speed"]["RMSE"],         errors_e[1e-3]["speed"]["corr"],         errors_e[1e-4]["speed"]["corr"] ),
    ]
    
    table = pd.DataFrame(
        data=table_data,
        index=names + ["h + z", "speed"],
        columns=["Quantity", "RMSE, \epsilon = 10-3", "RMSE, \epsilon = 10-4", "R2, \epsilon = 10-3", "R2, \epsilon = 10-4"]
    )
    
    table.to_csv("table.csv")
    

def compare_all_stage_timeseries(): 
    experimental_stage_timeseries = load_experimental_stage_timeseries()
    all_computed_stage_timeseries = load_all_computed_stage_timeseries()
    all_stage_point_elevations    = read_all_stage_point_elevations()
    #all_computed_maps             = load_all_computed_maps()
    
    #write_table(all_computed_stage_timeseries, all_computed_maps)
    
    fig = plt.figure( figsize=(6, 7) )
    
    gridspec = fig.add_gridspec(
        nrows=5,
        ncols=1,
        hspace=1.4
    )
    
    axs = gridspec.subplots()
    
    compare_stage_timeseries(ax=axs[0], experimental_stage_timeseries=experimental_stage_timeseries, all_computed_stage_timeseries=all_computed_stage_timeseries, all_stage_point_elevations=all_stage_point_elevations, name="A Beacon")
    compare_stage_timeseries(ax=axs[1], experimental_stage_timeseries=experimental_stage_timeseries, all_computed_stage_timeseries=all_computed_stage_timeseries, all_stage_point_elevations=all_stage_point_elevations, name="Tug Harbour")
    compare_stage_timeseries(ax=axs[2], experimental_stage_timeseries=experimental_stage_timeseries, all_computed_stage_timeseries=all_computed_stage_timeseries, all_stage_point_elevations=all_stage_point_elevations, name="Sulphur Point")
    compare_stage_timeseries(ax=axs[3], experimental_stage_timeseries=experimental_stage_timeseries, all_computed_stage_timeseries=all_computed_stage_timeseries, all_stage_point_elevations=all_stage_point_elevations, name="Moturiki")
    compare_stage_timeseries(ax=axs[4], experimental_stage_timeseries=experimental_stage_timeseries, all_computed_stage_timeseries=all_computed_stage_timeseries, all_stage_point_elevations=all_stage_point_elevations, name="Moturiki", speed=True)
    
    fig.savefig(fname="predictions.svg", bbox_inches="tight")

def compare_velocity_timeseries(
    ax
):
    experimental_velocity_timeseries = load_experimental_velocity_timeseries()
    all_computed_velocity_timeseries = load_all_computed_velocity_timeseries()
    
    for epsilon in all_computed_velocity_timeseries:
        computed_velocity_timeseries = all_computed_velocity_timeseries[epsilon]
        
        ax.plot(
            computed_velocity_timeseries["time"],
            computed_velocity_timeseries["speed"],
            linewidth=0.85
        )
        
    ax.plot(
        experimental_velocity_timeseries["time"],
        experimental_velocity_timeseries["speed"]["total"],
        linewidth=0.85,
        label="Experimental",
        color='k',
        zorder=3
    )
    
    ymin = -0.2
    ymax =  2.5
    
    num_yticks = 4
    
    dy = (ymax - ymin) / num_yticks
    
    yticks = [ round(ymin + dy * i, 1) for i in range(num_yticks+1) ]
    
    ax.set_yticks( [] )
    ax.set_yticks(
        ticks=yticks,
        minor=False
    )
    
    ax.set_yticklabels(
        labels=yticks,
        minor=False
    )
    
    xmin = 10
    xmax = 35
    
    num_xticks = 10
    
    dx = (xmax - xmin) / num_xticks
    
    xticks = [ round(xmin + dx * i, 1) for i in range(num_xticks+1) ]
    
    ax.set_xticks( [] )
    ax.set_xticks(
        ticks=xticks,
        minor=False
    )
    
    ax.set_xticklabels(
        labels=xticks,
        minor=False
    )
    
    plt.setp(
        ax,
        title="ADCP",
        xlim=(xmin,xmax),
        ylim=(ymin,ymax),
        xlabel="$t$ (hr)",
        ylabel="Speed (ms$^{-1}$)"
    )
    
    row_data = [
        np.sqrt( np.square( all_computed_velocity_timeseries[1e-3]["speed"] - all_computed_velocity_timeseries[0]["speed"] ).mean() ),
        np.sqrt( np.square( all_computed_velocity_timeseries[1e-4]["speed"] - all_computed_velocity_timeseries[0]["speed"] ).mean() ),
        np.corrcoef( x=all_computed_velocity_timeseries[1e-3]["speed"], y=all_computed_velocity_timeseries[0]["speed"] )[0][1],
        np.corrcoef( x=all_computed_velocity_timeseries[1e-4]["speed"], y=all_computed_velocity_timeseries[0]["speed"] )[0][1],
    ]
    
    with open("table.csv", 'a') as fp:
        fp.write(f"ACDP,speed,{row_data[0]},{row_data[1]},{row_data[2]},{row_data[3]}")

class YAxisLimits:
    def __init__(self):
        limits = {}
        
        limits["reduction"]   = {}
        limits["frac_DG2"]    = {}
        limits["rel_speedup"] = {}
        
        for key in limits:
            limits[key]["min"] =  (2 ** 63 + 1)
            limits[key]["max"] = -(2 ** 63 + 1)
        
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
        
def plot_speedups():
    print("Plotting speedups...")
    
    ref_runtime = np.loadtxt(
        fname=os.path.join("_eps-0", "res.cumu"),
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
    
    for dirroot, color in zip(dirroots, colors):
        if dirroot == "_eps-0": continue
        
        cumulative_data = np.loadtxt(fname=os.path.join(dirroot, "res.cumu"), skiprows=1, delimiter=',')
        
        time = cumulative_data[:,0] / 3600
        
        rel_speedup = ref_runtime / cumulative_data[:,3]
        
        reduction = 100 * cumulative_data[:,5]
        
        ax_rel_speedup.plot(
            time,
            rel_speedup,
            linewidth=2
        )
        
        y_axis_limits.set_y_axis_limits(field="rel_speedup", field_data=rel_speedup[1:])
        
        ax_rel_speedup.set_ylabel("$S_{rel}$ (-)")
        
        line, = ax_reduction.plot(
            time,
            reduction,
            linewidth=2,
            color=color
        )
        
        ax_reduction.plot(
            [ time[0], time[-1] ],
            [ reduction[0], reduction[0] ],
            linestyle='--',
            linewidth=1.5,
            color=color
        )
        
        y_axis_limits.set_y_axis_limits(field="reduction", field_data=reduction)
        
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
    
    #ax_reduction.invert_yaxis()
    
    y_axis_limits.set_y_axis_ticks(ax=ax_rel_speedup, field="rel_speedup", num_ticks=5, num_digits_round=1)
    y_axis_limits.set_y_axis_ticks(ax=ax_reduction,   field="reduction",   num_ticks=10)
    y_axis_limits.set_y_axis_ticks(ax=ax_frac_DG2,    field="frac_DG2",    num_ticks=5)
    
    xlim = ( 0, round(time[-1], 0) )
    
    for ax in axs:
        ax.set_xlim(xlim)
    
    ax_reduction.set_xlabel("$t$ (hr)")
    ax_rel_speedup.set_xlabel("$t$ (hr)")
    ax_reduction.legend(handles=lines, labels=["$\epsilon = 10^{-3}$", "$\epsilon = 10^{-4}$"])
    fig.tight_layout()
    fig.savefig(fname="speedups.svg", bbox_inches="tight")
    plt.close()

def main():
    compare_all_stage_timeseries()
    plot_speedups()
    
if __name__ == "__main__":
    main()