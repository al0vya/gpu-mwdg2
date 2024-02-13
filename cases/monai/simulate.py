import os
import sys
import subprocess
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import collections

def EXIT_HELP():
    help_message = ("Use this tool as:\n" + "python simulate.py <SOLVER>, SOLVER={hwfv1|mwdg2} to select either the GPU-HWFV1 or GPU-MWDG2 solver, respectively.\n")
    
    sys.exit(help_message)

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
        
class ExperimentalDataMonai:
    def __init__(self):
        self.gauge_data = {}
        
        print("Reading experimental data...")
        
        experimental_dataframe = pd.read_csv("MonaiValley_WaveGages.txt", delimiter="\t")
        
        self.time                  = experimental_dataframe["Time (sec)"]
        self.gauge_data["Point 1"] = experimental_dataframe["Gage 1 (cm)"]
        self.gauge_data["Point 2"] = experimental_dataframe["Gage 2 (cm)"]
        self.gauge_data["Point 3"] = experimental_dataframe["Gage 3 (cm)"]

class SimulationMonai:
    def __init__(
            self,
            solver
        ):
            print("Creating fields for simulation results...")
            
            self.solver      = solver
            self.epsilons    = [1e-3, 1e-4, 0]
            self.dirroots    = ["eps-1e-3", "eps-1e-4", "eps-0"]
            self.input_file  = "monai.par"
            self.points      = ["Point 1", "Point 2", "Point 3"]
            
            red_dd = lambda: collections.defaultdict(red_dd)
            
            self.results = red_dd()
            
            self.write_par_file()
            
            simulation_runs = 1
            
            for epsilon, dirroot_base in zip(self.epsilons, self.dirroots):
                for run in range(simulation_runs):
                    dirroot = dirroot_base + "-" + str(run)
                    
                    self.run(epsilon, dirroot)
                    
                    cumulative_dataframe = pd.read_csv( os.path.join(dirroot, "res.cumu") )
                    
                    self.results[run][epsilon]["simtime"]        = cumulative_dataframe["simtime"]
                    self.results[run][epsilon]["runtime_mra"]    = cumulative_dataframe["runtime_mra"]
                    self.results[run][epsilon]["runtime_solver"] = cumulative_dataframe["runtime_solver"]
                    self.results[run][epsilon]["runtime_total"]  = cumulative_dataframe["runtime_total"]
                    self.results[run][epsilon]["reduction"]      = cumulative_dataframe["reduction"]
                    
                    if run > 0:
                        continue
                    
                    gauge_dataframe = pd.read_csv(
                        os.path.join(dirroot, "res.stage"),
                        skiprows=9,
                        delimiter=" ",
                        header=None
                    )
                    
                    self.results[-1][epsilon]["gauge_data"]["Point 1"] = gauge_dataframe.iloc[:,1]
                    self.results[-1][epsilon]["gauge_data"]["Point 2"] = gauge_dataframe.iloc[:,2]
                    self.results[-1][epsilon]["gauge_data"]["Point 3"] = gauge_dataframe.iloc[:,3]
                    
                    self.results[-1][epsilon]["map"] = np.loadtxt(fname=os.path.join(dirroot, "res-1.elev"), skiprows=6)
            
            rows = self.results[0][0]["simtime"].shape[0]
            
            for epsilon in self.epsilons:
                self.results[-1][epsilon]["simtime"]        = np.zeros( shape=(rows,simulation_runs) )
                self.results[-1][epsilon]["runtime_mra"]    = np.zeros( shape=(rows,simulation_runs) )
                self.results[-1][epsilon]["runtime_solver"] = np.zeros( shape=(rows,simulation_runs) )
                self.results[-1][epsilon]["runtime_total"]  = np.zeros( shape=(rows,simulation_runs) )
                self.results[-1][epsilon]["reduction"]      = np.zeros( shape=(rows,simulation_runs) )
                
                for run in range(simulation_runs):
                    self.results[-1][epsilon]["simtime"][:,run]        = self.results[run][epsilon]["simtime"]       
                    self.results[-1][epsilon]["runtime_mra"][:,run]    = self.results[run][epsilon]["runtime_mra"]   
                    self.results[-1][epsilon]["runtime_solver"][:,run] = self.results[run][epsilon]["runtime_solver"]
                    self.results[-1][epsilon]["runtime_total"][:,run]  = self.results[run][epsilon]["runtime_total"] 
                    self.results[-1][epsilon]["reduction"] [:,run]     = self.results[run][epsilon]["reduction"]     
                
                self.results[-1][epsilon]["simtime"]        = self.results[-1][epsilon]["simtime"].mean(axis=1)       
                self.results[-1][epsilon]["runtime_mra"]    = self.results[-1][epsilon]["runtime_mra"].mean(axis=1)   
                self.results[-1][epsilon]["runtime_solver"] = self.results[-1][epsilon]["runtime_solver"].mean(axis=1)
                self.results[-1][epsilon]["runtime_total"]  = self.results[-1][epsilon]["runtime_total"].mean(axis=1) 
                self.results[-1][epsilon]["reduction"]      = self.results[-1][epsilon]["reduction"].mean(axis=1)     
    
    def write_par_file(self):
        with open(self.input_file, 'w') as fp:
            params = (
                f"{self.solver}\n" +
                "cuda\n" +
                "raster_out\n" +
                "cumulative\n" +
                "refine_wall\n" +
                "ref_thickness 16\n" +
                "max_ref_lvl   9\n" +
                "epsilon       0\n" +
                "wall_height   0.5\n" +
                "initial_tstep 1\n" +
                "fpfric        0.01\n" +
                "sim_time      22.5\n" +
                "massint       0.2\n" +
                "saveint       22.5\n" +
                "DEMfile       monai.dem\n" +
                "startfile     monai.start\n" +
                "bcifile       monai.bci\n" +
                "bdyfile       monai.bdy\n" +
                "stagefile     monai.stage\n"
            )
            
            fp.write(params)
    
    def run(
            self,
            epsilon,
            dirroot
        ):
            print("Running simulation, eps = " + str(epsilon) + ", solver: " + solver)
            
            executable = "gpu-mwdg2.exe" if sys.platform == "win32" else "gpu-mwdg2"
            
            command_line_args = [
                os.path.join("..", executable),
                "-epsilon", str(epsilon),
                "-dirroot", str(dirroot),
                self.input_file
            ]
            
            subprocess.run(command_line_args)
        
    def compute_root_mean_squared_errors(self):
        RMSE = [
            np.sqrt( np.square( self.results[-1][1e-3]["gauge_data"]["Point 1"].to_numpy() - self.results[-1][0]["gauge_data"]["Point 1"].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[-1][1e-4]["gauge_data"]["Point 1"].to_numpy() - self.results[-1][0]["gauge_data"]["Point 1"].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[-1][1e-3]["gauge_data"]["Point 2"].to_numpy() - self.results[-1][0]["gauge_data"]["Point 2"].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[-1][1e-4]["gauge_data"]["Point 2"].to_numpy() - self.results[-1][0]["gauge_data"]["Point 2"].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[-1][1e-3]["gauge_data"]["Point 3"].to_numpy() - self.results[-1][0]["gauge_data"]["Point 3"].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[-1][1e-4]["gauge_data"]["Point 3"].to_numpy() - self.results[-1][0]["gauge_data"]["Point 3"].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[-1][1e-3]["map"] - self.results[-1][0]["map"] ).mean() ),
            np.sqrt( np.square( self.results[-1][1e-4]["map"] - self.results[-1][0]["map"] ).mean() )
        ]
        
        return pd.DataFrame(
            [[RMSE[0], RMSE[1]], [RMSE[2], RMSE[3]], [RMSE[4], RMSE[5]], [RMSE[6], RMSE[7]]],
            [f"Time series at {point}" for point in self.points] + ["Map"],
            ["\epsilon = 10-3", "\epsilon = 10-4"]
        )
        
    def compute_correlation(self):
        corr = [
            np.corrcoef(x=self.results[-1][1e-3]["gauge_data"]["Point 1"].to_numpy(), y=self.results[-1][0]["gauge_data"]["Point 1"].to_numpy() )[0][1],
            np.corrcoef(x=self.results[-1][1e-4]["gauge_data"]["Point 1"].to_numpy(), y=self.results[-1][0]["gauge_data"]["Point 1"].to_numpy() )[0][1],
            np.corrcoef(x=self.results[-1][1e-3]["gauge_data"]["Point 2"].to_numpy(), y=self.results[-1][0]["gauge_data"]["Point 2"].to_numpy() )[0][1],
            np.corrcoef(x=self.results[-1][1e-4]["gauge_data"]["Point 2"].to_numpy(), y=self.results[-1][0]["gauge_data"]["Point 2"].to_numpy() )[0][1],
            np.corrcoef(x=self.results[-1][1e-3]["gauge_data"]["Point 3"].to_numpy(), y=self.results[-1][0]["gauge_data"]["Point 3"].to_numpy() )[0][1],
            np.corrcoef(x=self.results[-1][1e-4]["gauge_data"]["Point 3"].to_numpy(), y=self.results[-1][0]["gauge_data"]["Point 3"].to_numpy() )[0][1],
            np.corrcoef(x=self.results[-1][1e-3]["map"].flatten(), y=self.results[-1][0]["map"].flatten() )[0][1],
            np.corrcoef(x=self.results[-1][1e-4]["map"].flatten(), y=self.results[-1][0]["map"].flatten() )[0][1]
        ]
        
        return pd.DataFrame(
            [[corr[0], corr[1]], [corr[2], corr[3]], [corr[4], corr[5]], [corr[6], corr[7]]],
            [f"Time series at {point}" for point in self.points] + ["Map"],
            ["\epsilon = 10-3", "\epsilon = 10-4"]
        )
    
    def write_table(self):
        RMSE = self.compute_root_mean_squared_errors()
        corr = self.compute_correlation()
        
        table = pd.concat([RMSE, corr], axis=1, keys=["RMSE", "r"])
        
        table.to_csv("table.csv")
    
    def plot_exp_data(
        self,
        exp_data
    ):
        fig, axs = plt.subplots(
            figsize=(5,6),
            nrows=3,
            sharex=True
        )
        
        lines = []
        
        for epsilon in self.epsilons:
            line = axs[0].plot(
                self.results[-1][epsilon]["simtime"],
                self.results[-1][epsilon]["gauge_data"]["Point 1"] + 0.123591 - 0.13535,
                linewidth=2.5
            )[0]
            
            axs[1].plot(
                self.results[-1][epsilon]["simtime"],
                self.results[-1][epsilon]["gauge_data"]["Point 2"] + 0.132484 - 0.13535,
                linewidth=2.5
            )
            
            axs[2].plot(
                self.results[-1][epsilon]["simtime"],
                self.results[-1][epsilon]["gauge_data"]["Point 3"] + 0.130107 - 0.13535,
                linewidth=2.5
            )
            
            lines.append(line)
            
        line = axs[0].scatter(
            exp_data.time,
            exp_data.gauge_data["Point 1"] / 100, # convert from cm to m
            facecolor="None",
            edgecolor="black",
            s=5
        )
        
        lines.append(line)
        
        axs[1].scatter(
            exp_data.time,
            exp_data.gauge_data["Point 2"] / 100,
            facecolor="None",
            edgecolor="black",
            s=5
        )
        
        axs[2].scatter(
            exp_data.time,
            exp_data.gauge_data["Point 3"] / 100,
            facecolor="None",
            edgecolor="black",
            s=5
        )
        
        axs[2].set_xlabel("$t$ (s)")
        axs[2].set_xlim( (0,22.5) )
        
        axs[0].set_title("Point 1")
        axs[1].set_title("Point 2")
        axs[2].set_title("Point 3")
        
        for ax in axs:
            ax.set_ylim( (-0.02, 0.05) )
            ax.set_ylabel("$h + z$ (m)")
        
        main_labels = [
            ("GPU-MWDG2" if self.solver == "mwdg2" else "GPU-HWFV1") + ", $\epsilon = 10^{-3}$",
            ("GPU-MWDG2" if self.solver == "mwdg2" else "GPU-HWFV1") + ", $\epsilon = 10^{-4}$",
            "GPU-DG2"    if self.solver == "mwdg2" else "GPU-HWFV1",
            "Experimental"
        ]
        
        axs[0].legend(handles=lines, labels=main_labels, bbox_to_anchor=(1.0,2.0),ncol=2)
        
        fig.tight_layout()
        
        fig.savefig("predictions-" + self.solver, bbox_inches="tight")
            
    def plot_speedups(self):
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
        
        for epsilon, color in zip(self.epsilons, colors):
            if epsilon == 0:
                continue
            
            time = self.results[-1][epsilon]["simtime"]
            
            rel_speedup = self.results[-1][0]["runtime_total"] / self.results[-1][epsilon]["runtime_total"]
            
            compression = 100 * self.results[-1][epsilon]["reduction"]
            
            if   np.isclose(epsilon, 1e-3):
                label = "$\epsilon = 10^{-3}$"
            elif np.isclose(epsilon, 1e-4):
                label = "$\epsilon = 10^{-4}$"
            
            ax_rel_speedup.plot(
                time,
                rel_speedup,
                linewidth=2
            )
            
            y_axis_limits.set_y_axis_limits(field="rel_speedup", field_data=rel_speedup)
            
            ax_rel_speedup.set_ylabel("$S_{rel}$ (-)")
            
            ax_reduction.plot(
                time,
                compression,
                linewidth=2,
                color=color,
                label=label
            )
            
            ax_reduction.plot(
                [ time[0], time[-1] ],
                [ compression[0], compression[0] ],
                linestyle='--',
                linewidth=1.5,
                color=color
            )
            
            y_axis_limits.set_y_axis_limits(field="reduction", field_data=compression)
            
            ax_reduction.set_ylabel("$R_{cell}$ (%)")
            
            frac_DG2 = 100 * (
                self.results[-1][epsilon]["runtime_solver"]
                /
                self.results[-1][epsilon]["runtime_total"]
            )
            
            ax_frac_DG2.plot(
                time[1:],
                frac_DG2[1:],
                linewidth=2
            )
            
            y_axis_limits.set_y_axis_limits(field="frac_DG2", field_data=frac_DG2[1:])
            
            ax_frac_DG2.set_ylabel("$F_{DG2}$ (%)")
        
        y_axis_limits.set_y_axis_ticks(ax=ax_rel_speedup, field="rel_speedup", num_ticks=5, num_digits_round=1)
        y_axis_limits.set_y_axis_ticks(ax=ax_reduction,   field="reduction",   num_ticks=10)
        y_axis_limits.set_y_axis_ticks(ax=ax_frac_DG2,    field="frac_DG2",    num_ticks=5)
        
        xlim = (
            0,
            round(self.results[-1][0]["simtime"][-1], 1)
        )
        
        for ax in axs:
            ax.set_xlim(xlim)
            
        ax_reduction.set_xlabel("$t$ (s)")
        ax_rel_speedup.set_xlabel("$t$ (s)")
        
        ax_reduction.legend()
        
        fig.tight_layout()
        
        fig.savefig("speedups-" + self.solver, bbox_inches="tight")
        
    def plot(
        self,
        exp_data
    ):
        self.plot_speedups()
        self.plot_exp_data(exp_data)
        self.write_table()
        
if __name__ == "__main__":
    if len(sys.argv) != 2:
        EXIT_HELP()
        
    dummy, solver = sys.argv
    
    if solver != "hwfv1" and solver != "mwdg2":
        EXIT_HELP()
    
    #subprocess.run( ["python", "stage.py" ] )
    #subprocess.run( ["python", "inflow.py"] )
    #subprocess.run( ["python", "raster.py"] )
    
    SimulationMonai(solver).plot( ExperimentalDataMonai() )