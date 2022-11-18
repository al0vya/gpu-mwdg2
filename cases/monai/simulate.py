import os
import sys
import subprocess
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

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
        num_ticks,
        num_digits_round
    ):
        if field == "rel_speedup":
            min_val = 0.9 * self.limits[field]["min"]
            max_val = 1.1 * self.limits[field]["max"]
        else:
            min_val = max( 0,   0.9 * self.limits[field]["min"] )
            max_val = min( 100, 1.1 * self.limits[field]["max"] )
        
        d_val = (max_val - min_val) / num_ticks
        
        if field == "rel_speedup":
            return [ round(min_val + i * d_val, num_digits_round) for i in range(num_ticks+1) ]
        else:
            return [ int( round(min_val + i * d_val, num_digits_round) ) for i in range(num_ticks+1) ]
        
    def set_y_axis_ticks(
        self,
        ax,
        field,
        num_ticks,
        num_digits_round=1
    ):
        yticks = self.get_y_axis_ticks(
            field=field,
            num_ticks=num_ticks,
            num_digits_round=num_digits_round
        )
        
        ax.set_yticks( [] )
        ax.set_yticks(
            ticks=yticks,
            minor=False
        )
        
        ax.set_yticklabels(
            labels=yticks,
            minor=False
        )
        
class ExperimentalDataMonai:
    def __init__(self):
        print("Reading experimental data...")
        
        experimental_dataframe = pd.read_csv("experimental.txt")
        
        self.time       = experimental_dataframe["time"]
        self.gauge_data = experimental_dataframe["stage1"]

class SimulationMonai:
    def __init__(
            self,
            epsilons,
            solvers
        ):
            print("Creating fields for simulation results...")
            
            self.solvers      = solvers
            self.epsilons     = epsilons
            self.fields       = [
                "simtime",
                "runtime_mra",
                "runtime_solver",
                "runtime_total",
                "compression",
                "gauge_data"
            ]
            self.stage_file   = os.path.join("results", "stage.wd")
            self.runtime_file = os.path.join("results", "cumulative-data.csv")
            self.results      = {}
            
            for solver in self.solvers:
                self.results[solver] = {}
                
                for epsilon in self.epsilons:
                    if epsilon not in self.results:
                        self.results[solver][epsilon] = {}
                        for field in self.fields:
                            if field not in self.results[solver][epsilon]:
                                self.results[solver][epsilon][field] = {}
                                
                for epsilon in epsilons:
                    self.run(epsilon, solver)
                    
                    cumulative_dataframe = pd.read_csv(self.runtime_file)
                    
                    self.results[solver][epsilon]["simtime"]        = cumulative_dataframe["simtime"]
                    self.results[solver][epsilon]["runtime_mra"]    = cumulative_dataframe["runtime_mra"]
                    self.results[solver][epsilon]["runtime_solver"] = cumulative_dataframe["runtime_solver"]
                    self.results[solver][epsilon]["runtime_total"]  = cumulative_dataframe["runtime_total"]
                    self.results[solver][epsilon]["compression"]    = cumulative_dataframe["compression"]
                    self.results[solver][epsilon]["gauge_data"]     = pd.read_csv(
                        self.stage_file,
                        skiprows=7,
                        delimiter=" ",
                        header=None
                    ).iloc[:,1]
            
    def run(
            self,
            epsilon,
            solver
        ):
            print("Running simulation, eps = " + str(epsilon) + ", solver: " + solver)
            
            input_file = "monai.par"
            
            with open(input_file, 'w') as fp:
                params = (
                    "test_case     0\n" +
                    "max_ref_lvl   9\n" +
                    "min_dt        1\n" +
                    "respath       results\n" +
                    "epsilon       %s\n" +
                    "fpfric        0.01\n" +
                    "rasterroot    monai\n" +
                    "bcifile       monai.bci\n" +
                    "bdyfile       monai.bdy\n" +
                    "stagefile     monai.stage\n" +
                    "tol_h         1e-3\n" +
                    "tol_q         0\n" +
                    "tol_s         1e-9\n" +
                    "g             9.80665\n" +
                    "massint       0.2\n" +
                    "sim_time      22.5\n" +
                    "solver        %s\n" +
                    "limitslopes   off\n" +
                    "tol_Krivo     10\n" +
                    "refine_wall   on\n" +
                    "ref_thickness 16\n" +
                    "cumulative    on\n" +
                    "wall_height   0.5"
                ) % (
                    epsilon,
                    solver
                )
                
                fp.write(params)
            
            executable = "gpu-mwdg2.exe" if sys.platform == "win32" else "gpu-mwdg2"
            
            subprocess.run( [os.path.join("..", executable), input_file] )
            
    def compute_instant_runtime_ratio(
        self,
        runtime_uniform,
        runtime_adapt
    ):
        instant_runtime_ratio = (
            ( runtime_uniform[1:] - runtime_uniform[:-1] )
            /
            ( runtime_adapt[1:] - runtime_adapt[:-1] )
        )
        
        return np.append( instant_runtime_ratio, instant_runtime_ratio[-1] )
       
    def plot_exp_data(
        self,
        my_rc_params,
        exp_data
    ):
        plt.rcParams.update(my_rc_params)
        
        fig, ax = plt.subplots( figsize=(2.75, 2.5) )
        
        for solver in self.solvers:
            for epsilon in self.epsilons:
                if epsilon == 0:
                    label = "GPU-DG2"    if solver == "mw" else "GPU-FV1"
                elif np.isclose(epsilon, 1e-3):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-3}$"
                elif np.isclose(epsilon, 1e-4):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-4}$"

                ax.plot(
                    self.results[solver][epsilon]["simtime"],
                    self.results[solver][epsilon]["gauge_data"],
                    linewidth=2.5,
                    label=label
                )
            
        ax.scatter(
            exp_data.time,
            exp_data.gauge_data,
            facecolor="None",
            edgecolor="black",
            s=5,
            label="Experimental"
        )
        
        ax.set_xlabel(r"$t \, (s)$")
        ax.set_ylabel(r"h + z \, $(m)$")
        ax.set_xlim( exp_data.time.iloc[0], exp_data.time.iloc[-1] )
        ax.legend(
            bbox_to_anchor=(1.2, 1.35),
            ncol=2
        )
        fig.savefig(os.path.join("results", "stage"), bbox_inches="tight")
        plt.close()
            
    def plot_speedups(
        self,
        my_rc_params
    ):
        #plt.rcParams.DG2(my_rc_params)
        
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
        
        for solver in self.solvers:
            for epsilon, color in zip(self.epsilons, colors):
                if epsilon == 0:
                    continue
                
                time = self.results[solver][epsilon]["simtime"]
                
                rel_speedup = self.results[solver][0]["runtime_total"] / self.results[solver][epsilon]["runtime_total"]
                
                compression = 100 * self.results[solver][epsilon]["compression"]
                
                if   np.isclose(epsilon, 1e-3):
                    label = "$\epsilon = 10^{-3}$"
                elif np.isclose(epsilon, 1e-4):
                    label = "$\epsilon = 10^{-4}$"
                
                ax_rel_speedup.plot(
                    time,
                    rel_speedup,
                    linestyle='--',
                    linewidth=1.5
                )
                
                y_axis_limits.set_y_axis_limits(field="rel_speedup", field_data=rel_speedup)
                
                ax_rel_speedup.set_ylabel("$S_{rel}$ (-)")
                
                ax_reduction.plot(
                    time,
                    compression,
                    linestyle='--',
                    linewidth=1.5,
                    color=color,
                    label=label
                )
                
                ax_reduction.plot(
                    [ time.iloc[0], time.iloc[-1] ],
                    [ compression.iloc[0], compression.iloc[0] ],
                    linewidth=2,
                    color=color
                )
                
                y_axis_limits.set_y_axis_limits(field="reduction", field_data=compression)
                
                ax_reduction.set_ylabel("Reduction (%)")
                
                frac_DG2 = 100 * (
                    self.results[solver][epsilon]["runtime_solver"]
                    /
                    self.results[solver][epsilon]["runtime_total"]
                )
                
                ax_frac_DG2.plot(
                    time[1:],
                    frac_DG2[1:],
                    linestyle='--',
                    linewidth=1.5
                )
                
                y_axis_limits.set_y_axis_limits(field="frac_DG2", field_data=frac_DG2[1:])
                
                ax_frac_DG2.set_ylabel("$F_{DG2}$ (%)")
            
            ax_reduction.invert_yaxis()
            
            y_axis_limits.set_y_axis_ticks(ax=ax_rel_speedup, field="rel_speedup", num_ticks=5, num_digits_round=1)
            y_axis_limits.set_y_axis_ticks(ax=ax_reduction,   field="reduction",   num_ticks=10)
            y_axis_limits.set_y_axis_ticks(ax=ax_frac_DG2,    field="frac_DG2",    num_ticks=5)
            
            xlim = (
                0,
                round(self.results[solver][0]["simtime"].iloc[-1], 1)
            )
            
            for ax in axs:
                ax.set_xlim(xlim)
                
            ax_reduction.set_xlabel("$t$ (s)")
            ax_rel_speedup.set_xlabel("$t$ (s)")
            ax_reduction.legend()
            fig.tight_layout()
            fig.savefig(os.path.join("results", "runtimes-" + solver), bbox_inches="tight")
            ax.clear()
                
        plt.close()
    
    def plot(
        self,
        exp_data
    ):
        my_rc_params = {
            "legend.fontsize" : "small"
        }
        
        self.plot_speedups(my_rc_params)
        self.plot_exp_data(my_rc_params, exp_data)
        
if __name__ == "__main__":
    subprocess.run( ["python", "stage.py" ] )
    subprocess.run( ["python", "inflow.py"] )
    subprocess.run( ["python", "raster.py"] )
    
    SimulationMonai( [1e-3, 1e-4, 0], ["mw"] ).plot( ExperimentalDataMonai() )