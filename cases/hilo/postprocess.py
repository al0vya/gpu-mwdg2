# This script is used to postprocess results and plot graphs for the Hilo harbour
# test case at coastal.usc.edu/currents_workshop/problems/prob2.html

import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
def plot_predictions():
    print("Plotting predictions...")
    
    control_point_data = np.loadtxt( fname=os.path.join("input-data", "se.txt") )
    time_control_point = control_point_data[:,0] / 60
    eta_control_point_no_tide = control_point_data[:,1]-control_point_data[0,1]
    
    tide_gauge_data = np.loadtxt( fname=os.path.join("input-data", "TG_1617760_detided.txt") )
    time_tide_gauge = tide_gauge_data[:,0] / 3600
    eta_tide_gauge  = tide_gauge_data[:,1]
    
    timeshift = time_control_point[0]
    
    eta_predictions_0    = np.loadtxt(fname=os.path.join("eps-0",    "res.stage"), skiprows=11, delimiter=' ')
    eta_predictions_1e_3 = np.loadtxt(fname=os.path.join("eps-1e-3", "res.stage"), skiprows=11, delimiter=' ')
    eta_predictions_1e_4 = np.loadtxt(fname=os.path.join("eps-1e-4", "res.stage"), skiprows=11, delimiter=' ')
    
    control_point_0    = eta_predictions_0[:,2]    - 30
    control_point_1e_3 = eta_predictions_1e_3[:,2] - 30
    control_point_1e_4 = eta_predictions_1e_4[:,2] - 30
    
    tide_gauge_0    = eta_predictions_0[:,3]    + 19.546045 - 30
    tide_gauge_1e_3 = eta_predictions_1e_3[:,3] + 19.546045 - 30
    tide_gauge_1e_4 = eta_predictions_1e_4[:,3] + 19.546045 - 30
    
    vel_ha1125 = np.loadtxt( fname=os.path.join("input-data", "HAI1125_detided_harmonic.txt") )
    vel_ha1126 = np.loadtxt( fname=os.path.join("input-data", "HAI1126_detided_harmonic.txt") )
    
    ha1125_vy_0    = np.loadtxt(fname=os.path.join("eps-0",    "res.yvelocity"), skiprows=11, delimiter=' ')[:,4]
    ha1125_vy_1e_3 = np.loadtxt(fname=os.path.join("eps-1e-3", "res.yvelocity"), skiprows=11, delimiter=' ')[:,4]
    ha1125_vy_1e_4 = np.loadtxt(fname=os.path.join("eps-1e-4", "res.yvelocity"), skiprows=11, delimiter=' ')[:,4]
    
    ha1126_vx_0    = np.loadtxt(fname=os.path.join("eps-0",    "res.xvelocity"), skiprows=11, delimiter=' ')[:,5]
    ha1126_vx_1e_3 = np.loadtxt(fname=os.path.join("eps-1e-3", "res.xvelocity"), skiprows=11, delimiter=' ')[:,5]
    ha1126_vx_1e_4 = np.loadtxt(fname=os.path.join("eps-1e-4", "res.xvelocity"), skiprows=11, delimiter=' ')[:,5]
    
    map_el_0    = np.loadtxt(fname=os.path.join("eps-0",    "res-1.elev"), skiprows=6)
    map_el_1e_3 = np.loadtxt(fname=os.path.join("eps-1e-3", "res-1.elev"), skiprows=6)
    map_el_1e_4 = np.loadtxt(fname=os.path.join("eps-1e-4", "res-1.elev"), skiprows=6)
    
    map_vx_0    = np.loadtxt(fname=os.path.join("eps-0",    "res-1.vx"), skiprows=6)
    map_vx_1e_3 = np.loadtxt(fname=os.path.join("eps-1e-3", "res-1.vx"), skiprows=6)
    map_vx_1e_4 = np.loadtxt(fname=os.path.join("eps-1e-4", "res-1.vx"), skiprows=6)
    
    map_vy_0    = np.loadtxt(fname=os.path.join("eps-0",    "res-1.vy"), skiprows=6)
    map_vy_1e_3 = np.loadtxt(fname=os.path.join("eps-1e-3", "res-1.vy"), skiprows=6)
    map_vy_1e_4 = np.loadtxt(fname=os.path.join("eps-1e-4", "res-1.vy"), skiprows=6)
        
    time = eta_predictions_1e_3[:,0]/3600 + timeshift
    
    timeslice = (time > 8.5) & (time < 11)
    
    rec_dd = lambda: collections.defaultdict(rec_dd)
    
    table_data = rec_dd()

    table_data[1e-3]["cp"]["RMSE"] = np.sqrt( np.square( control_point_1e_3[timeslice] - control_point_0[timeslice]).mean() )
    table_data[1e-4]["cp"]["RMSE"] = np.sqrt( np.square( control_point_1e_4[timeslice] - control_point_0[timeslice]).mean() )
    table_data[1e-3]["tg"]["RMSE"] = np.sqrt( np.square( tide_gauge_1e_3[timeslice]    - tide_gauge_0[timeslice]   ).mean() )
    table_data[1e-4]["tg"]["RMSE"] = np.sqrt( np.square( tide_gauge_1e_4[timeslice]    - tide_gauge_0[timeslice]   ).mean() )
    table_data[1e-3]["25"]["RMSE"] = np.sqrt( np.square( ha1125_vy_1e_3[timeslice]     - ha1125_vy_0[timeslice]    ).mean() )
    table_data[1e-4]["25"]["RMSE"] = np.sqrt( np.square( ha1125_vy_1e_4[timeslice]     - ha1125_vy_0[timeslice]    ).mean() )
    table_data[1e-3]["26"]["RMSE"] = np.sqrt( np.square( ha1126_vx_1e_3[timeslice]     - ha1126_vx_0[timeslice]    ).mean() )
    table_data[1e-4]["26"]["RMSE"] = np.sqrt( np.square( ha1126_vx_1e_4[timeslice]     - ha1126_vx_0[timeslice]    ).mean() )
    table_data[1e-3]["el"]["RMSE"] = np.sqrt( np.square( map_el_1e_3                   - map_el_0                  ).mean() )
    table_data[1e-4]["el"]["RMSE"] = np.sqrt( np.square( map_el_1e_4                   - map_el_0                  ).mean() )
    table_data[1e-3]["vx"]["RMSE"] = np.sqrt( np.square( map_vx_1e_3                   - map_vx_0                  ).mean() )
    table_data[1e-4]["vx"]["RMSE"] = np.sqrt( np.square( map_vx_1e_4                   - map_vx_0                  ).mean() )
    table_data[1e-3]["vy"]["RMSE"] = np.sqrt( np.square( map_vy_1e_3                   - map_vy_0                  ).mean() )
    table_data[1e-4]["vy"]["RMSE"] = np.sqrt( np.square( map_vy_1e_4                   - map_vy_0                  ).mean() )
    
    table_data[1e-3]["cp"]["corr"] = np.corrcoef( x=control_point_1e_3[timeslice], y=control_point_0[timeslice] )[0][1]
    table_data[1e-4]["cp"]["corr"] = np.corrcoef( x=control_point_1e_4[timeslice], y=control_point_0[timeslice] )[0][1]
    table_data[1e-3]["tg"]["corr"] = np.corrcoef( x=tide_gauge_1e_3[timeslice]   , y=tide_gauge_0[timeslice]    )[0][1]
    table_data[1e-4]["tg"]["corr"] = np.corrcoef( x=tide_gauge_1e_4[timeslice]   , y=tide_gauge_0[timeslice]    )[0][1]
    table_data[1e-3]["25"]["corr"] = np.corrcoef( x=ha1125_vy_1e_3[timeslice]    , y=ha1125_vy_0[timeslice]     )[0][1]
    table_data[1e-4]["25"]["corr"] = np.corrcoef( x=ha1125_vy_1e_4[timeslice]    , y=ha1125_vy_0[timeslice]     )[0][1]
    table_data[1e-3]["26"]["corr"] = np.corrcoef( x=ha1126_vx_1e_3[timeslice]    , y=ha1126_vx_0[timeslice]     )[0][1]
    table_data[1e-4]["26"]["corr"] = np.corrcoef( x=ha1126_vx_1e_4[timeslice]    , y=ha1126_vx_0[timeslice]     )[0][1]        
    table_data[1e-3]["el"]["corr"] = np.corrcoef( x=map_el_1e_3.flatten()        , y=map_el_0.flatten()         )[0][1]        
    table_data[1e-4]["el"]["corr"] = np.corrcoef( x=map_el_1e_4.flatten()        , y=map_el_0.flatten()         )[0][1]        
    table_data[1e-3]["vx"]["corr"] = np.corrcoef( x=map_vx_1e_3.flatten()        , y=map_vx_0.flatten()         )[0][1]        
    table_data[1e-4]["vx"]["corr"] = np.corrcoef( x=map_vx_1e_4.flatten()        , y=map_vx_0.flatten()         )[0][1]        
    table_data[1e-3]["vy"]["corr"] = np.corrcoef( x=map_vy_1e_3.flatten()        , y=map_vy_0.flatten()         )[0][1]        
    table_data[1e-4]["vy"]["corr"] = np.corrcoef( x=map_vy_1e_4.flatten()        , y=map_vy_0.flatten()         )[0][1]        
    
    table = [
        ("h + z", table_data[1e-3]["cp"]["RMSE"], table_data[1e-4]["cp"]["RMSE"], table_data[1e-3]["cp"]["corr"], table_data[1e-4]["cp"]["corr"]),
        ("h + z", table_data[1e-3]["tg"]["RMSE"], table_data[1e-4]["tg"]["RMSE"], table_data[1e-3]["tg"]["corr"], table_data[1e-4]["tg"]["corr"]),
        ("vy",    table_data[1e-3]["25"]["RMSE"], table_data[1e-4]["25"]["RMSE"], table_data[1e-3]["25"]["corr"], table_data[1e-4]["25"]["corr"]),
        ("vx",    table_data[1e-3]["26"]["RMSE"], table_data[1e-4]["26"]["RMSE"], table_data[1e-3]["26"]["corr"], table_data[1e-4]["26"]["corr"]),
        ("h + z", table_data[1e-3]["el"]["RMSE"], table_data[1e-4]["el"]["RMSE"], table_data[1e-3]["el"]["corr"], table_data[1e-4]["el"]["corr"]),
        ("vx",    table_data[1e-3]["vx"]["RMSE"], table_data[1e-4]["vx"]["RMSE"], table_data[1e-3]["vx"]["corr"], table_data[1e-4]["vx"]["corr"]),
        ("vy",    table_data[1e-3]["vy"]["RMSE"], table_data[1e-4]["vy"]["RMSE"], table_data[1e-3]["vy"]["corr"], table_data[1e-4]["vy"]["corr"])
    ]
    
    table_df = pd.DataFrame(
        data=table,
        index=["Control point", "Tide gauge", "ADCP HA1125", "ADCP HA1126", "Spatial map at ts", "Spatial map at ts", "Spatial map at ts"],
        columns=["Quantity", "RMSE, \epsilon = 10-3", "RMSE, \epsilon = 10-4", "R2, \epsilon = 10-3", "R2, \epsilon = 10-4"]
    )
    
    table_df.to_csv("table.csv")
    
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
    
    lines.append( axs[0,0].plot(time, control_point_1e_3, label="GPU-MWDG2, $\epsilon = 10^{-3}$")[0] )
    lines.append( axs[0,0].plot(time, control_point_1e_4, label="GPU-MWDG2, $\epsilon = 10^{-4}$")[0] )
    lines.append( axs[0,0].plot(time, control_point_0,    label="GPU-DG2")                        [0] )
    
    lines.append( axs[0,0].plot(time_control_point, eta_control_point_no_tide, label="Experimental", color='k')[0] )
    
    axs[0,0].set_xlim( (8.5,11) )
    axs[0,0].set_ylim( (-1.5,1.7) )
    
    axs[0,0].set_title("Control point", fontsize="medium")
    axs[0,0].set_ylabel("$h + z$ (m)")
    
    axs[0,1].plot(time, tide_gauge_1e_3)
    axs[0,1].plot(time, tide_gauge_1e_4)
    axs[0,1].plot(time, tide_gauge_0)
    
    axs[0,1].plot(time_tide_gauge, eta_tide_gauge, label="Experimental", color='k')
    
    axs[0,1].set_xlim( (8.5,11) )
    axs[0,1].set_ylim( (-2.5,2.5) )
    
    axs[0,1].set_title("Tide gauge", fontsize="medium")
    axs[0,1].set_ylabel("$h + z$ (m)")
    
    axs[1,0].plot(time, ha1125_vy_1e_3)
    axs[1,0].plot(time, ha1125_vy_1e_4)
    axs[1,0].plot(time, ha1125_vy_0)
    
    axs[1,0].plot(vel_ha1125[:,0], vel_ha1125[:,2] / 100, label="Experimental", color='k')
    
    axs[1,0].set_xlim( (8.5,11) )
    axs[1,0].set_ylim( (-1.4,1.6) )
    
    axs[1,0].set_title("ADCP HA1125", fontsize="medium")
    axs[1,0].set_ylabel("$v$ (m/s)")
    
    axs[1,1].plot(time, ha1126_vx_1e_3)
    axs[1,1].plot(time, ha1126_vx_1e_4)
    axs[1,1].plot(time, ha1126_vx_0)
    
    axs[1,1].plot(vel_ha1126[:,0], vel_ha1126[:,1] / 100, label="Experimental", color='k')
    
    axs[1,1].set_xlim( (8.5,11) )
    axs[1,1].set_ylim( (-1.5,1.1) )

    axs[1,1].set_title("ADCP HA1126", fontsize="medium")
    axs[1,1].set_ylabel("$u$ (m/s)")
    
    main_labels = [
        "GPU-MWDG2, $\epsilon = 10^{-3}$",
        "GPU-MWDG2, $\epsilon = 10^{-4}$",
        "GPU-DG2",
        "Experimental"
    ]
    
    axs[0,0].legend(
        handles=lines,
        labels=main_labels,
        bbox_to_anchor=(1.7,1.55),
        ncol=2,
        fontsize="x-small"
    )
    
    fig.savefig("predictions.svg", bbox_inches="tight")
    
class YAxisLimits:
    def __init__(self):
        limits = {}
        
        limits["reduction"]   = {}
        limits["frac_DG2"]    = {}
        limits["rel_speedup"] = {}
        
        for key in limits:
            limits[key]["min"] =  (2 ** 63 + 1)
            limits[key]["max"] = -(2 ** 63 + 1)
        
        #limits["reduction"]["min"] = 0
        
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
    
    cumu_files = [
        os.path.join("eps-1e-3", "res.cumu"),
        os.path.join("eps-1e-4", "res.cumu"),
        os.path.join("eps-0",    "res.cumu")
    ]
    
    ref_runtime = np.loadtxt(
        fname=cumu_files[-1],
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
    
    for cumu_file, color in zip(cumu_files, colors):
        if cumu_file == os.path.join("eps-0", "res.cumu"):
            continue
        
        cumulative_data = np.loadtxt(fname=cumu_file, skiprows=1, delimiter=',')
        
        time = cumulative_data[:,0] / 3600 + 7 # convert from seconds to hours and shift by 7 hours
        
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
            [ time[0], round(time[-1], 0) ],
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
    
    y_axis_limits.set_y_axis_ticks(ax=ax_rel_speedup, field="rel_speedup", num_ticks=5, num_digits_round=1)
    y_axis_limits.set_y_axis_ticks(ax=ax_reduction,   field="reduction",   num_ticks=10)
    y_axis_limits.set_y_axis_ticks(ax=ax_frac_DG2,    field="frac_DG2",    num_ticks=5)
    
    for ax in axs:
        ax.set_xlim( (7,13) )
    
    ax_reduction.set_xlabel("$t$ (hr)")
    ax_rel_speedup.set_xlabel("$t$ (hr)")
    ax_reduction.legend(handles=lines, labels=["$\epsilon = 10^{-3}$", "$\epsilon = 10^{-4}$"])
    fig.tight_layout()
    fig.savefig(fname="speedups.svg", bbox_inches="tight")
    plt.close()

def main():
    plot_speedups()
    plot_predictions()
    
if __name__ == "__main__":
    main()