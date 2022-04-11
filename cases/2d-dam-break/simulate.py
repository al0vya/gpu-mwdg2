import os
import sys
import subprocess
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

class Simulation2DDambreak:
    def __init__(
        self,
        solvers
    ):
        self.solvers      = solvers
        self.results      = {}
        self.epsilons     = [0, 1e-4, 1e-3, 1e-2]
        self.fields       = ["simtime", "runtime"]
        self.max_ref_lvls = [8, 9, 10]#, 11]
        
        for solver in self.solvers:
            self.results[solver] = {}
            for epsilon in self.epsilons:
                self.results[solver][epsilon]          = {}
                self.results[solver][epsilon]["depth"] = {}
                for L in self.max_ref_lvls:
                    self.results[solver][epsilon][L] = {}
                    for field in self.fields:
                        self.results[solver][epsilon][L][field] = {}
        
        self.results["x"] = {}
                
        # runs for speedups
        for solver in self.solvers:
            for epsilon in self.epsilons:
                for L in self.max_ref_lvls:
                    self.run(
                        solver=solver,
                        sim_time=3.5,
                        epsilon=epsilon,
                        L=L,
                        saveint=3.5
                    )
                    
                    # for verification
                    if L == 8:
                        verification_depths = self.get_verification_depths()
                        
                        self.results["x"]                       = verification_depths["x"]
                        self.results[solver][epsilon]["depths"] = verification_depths["depths"]
                    
                    results_dataframe = pd.read_csv( os.path.join("results", "simtime-vs-runtime.csv") )
                    
                    self.results[solver][epsilon][L]["runtime"] = results_dataframe["runtime"].iloc[-1]
                    
    def get_verification_depths(
        self
    ):
        depths_frame    = pd.read_csv( os.path.join("results", "depths-1.csv") )
        mesh_info_frame = pd.read_csv( os.path.join("results", "mesh_info.csv") )
        
        xmin     = mesh_info_frame["xmin"].iloc[0]
        xmax     = mesh_info_frame["xmax"].iloc[0]
        mesh_dim = mesh_info_frame["mesh_dim"].iloc[0]
        
        dx = (xmax - xmin) / mesh_dim
        
        x = [ xmin + i * dx for i in range(mesh_dim) ]
        
        x[-1] += dx
        
        beg = int( (mesh_dim / 2) * mesh_dim )
        end = int( (mesh_dim / 2) * mesh_dim + mesh_dim)
        
        depths = depths_frame[beg:end]
        
        return {"x" : x, "depths" : depths}
    
    def run(
        self,
        solver,
        sim_time,
        epsilon,
        L,
        saveint
    ):
        print("Running simulation, L = " + str(L) + ", eps = " + str(epsilon) + ", solver: " + solver)
            
        test_script = os.path.join("..", "tests", "test.py")
        
        # using test.py to run simulations
        subprocess.run(
            [
                "python",
                test_script,   # test.py
                "run",         # ACTION
                solver,        # SOLVER
                "23",          # TEST_CASE
                str(sim_time), # SIM_TIME
                str(epsilon),  # EPSILON
                str(L),        # MAX_REF_LVL
                str(saveint),  # SAVE_INT
                str(sim_time), # MASS_INT
                "surf",        # PLOT_TYPE
                "off"          # SLOPE_LIMITER
            ]
        )
        
    def plot_speedups(
        self,
        my_rc_params
    ):
        plt.rcParams.update(my_rc_params)
        
        fig, ax = plt.subplots()
        
        for solver in self.solvers:
            for epsilon in self.epsilons:
                if epsilon == 0:
                    label = "breakeven"
                elif np.isclose(epsilon, 1e-2):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-2}$"
                elif np.isclose(epsilon, 1e-3):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-3}$"
                elif np.isclose(epsilon, 1e-4):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-4}$"
                
                speedups = [
                    self.results[solver][0][L]["runtime"] / self.results[solver][epsilon][L]["runtime"]
                    for L in self.max_ref_lvls
                ]
                
                ax.plot(
                    self.max_ref_lvls,
                    speedups,
                    marker=None    if epsilon == 0 else 'x',
                    linewidth=1    if epsilon == 0 else 0.75,
                    linestyle="-." if epsilon == 0 else '--',
                    label=label
                )
            
            xlim = ( self.max_ref_lvls[0], self.max_ref_lvls[-1] )
            
            ax.set_xlabel(r"$L$")
            ax.set_ylabel( "Speedup ratio " + ("GPU-MWDG2/GPU-DG2" if solver == "mw" else "GPU-HWFV1/GPU-FV1") )
            ax.set_xlim(xlim)
            ax.legend()
            fig.savefig(os.path.join("results", "runtimes-" + solver + ".png"), bbox_inches="tight")
            ax.clear()
            
        plt.close()
    
    def plot_verification_depths(
        self,
        my_rc_params
    ):
        plt.rcParams.update(my_rc_params)
        
        fig, ax = plt.subplots()
        
        for solver in self.solvers:
            for epsilon in self.epsilons:
                if epsilon == 0:
                    label = ("GPU-DG2"   if solver == "mw" else "GPU-FV1")
                elif np.isclose(epsilon, 1e-2):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-2}$"
                elif np.isclose(epsilon, 1e-3):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-3}$"
                elif np.isclose(epsilon, 1e-4):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-4}$"
                
                ax.plot(
                    self.results["x"],
                    self.results[solver][epsilon]["depths"],
                    label=label
                )
            
        xlim = ( self.results["x"][0], self.results["x"][-1] )
        
        ax.set_xlabel(r"$x \, (m)$")
        ax.set_ylabel(r"$h \, (m)$")
        ax.set_xlim(xlim)
        ax.legend()
        fig.savefig(os.path.join("results", "verification.png"), bbox_inches="tight")
        
        plt.close()
        
    def plot(self):
        my_rc_params = {
            "legend.fontsize" : "large",
            "axes.labelsize"  : "xx-large",
            "axes.titlesize"  : "xx-large",
            "xtick.labelsize" : "xx-large",
            "ytick.labelsize" : "xx-large",
        }
        
        self.plot_speedups(my_rc_params)
        
        self.plot_verification_depths(my_rc_params)
        
if __name__ == "__main__":
    Simulation2DDambreak( ["mw"] ).plot()