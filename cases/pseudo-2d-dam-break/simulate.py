import os
import sys
import subprocess
import scipy.interpolate
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

def calc_cm(x, c_left, c_right):
    return x ** 6 - 9 * c_right ** 2 * x ** 4 + 16 * c_left * c_right ** 2 *x ** 3 - c_right ** 2 * (c_right ** 2 + 8 * c_left ** 2) * x** 2 + c_right ** 6

def calc_h_exact(xdam, x, Lx, h_left, h_right, t):
    g = 9.81
    
    c_left  = np.sqrt(g * h_left)
    c_right = np.sqrt(g * h_right)
    
    eps = 0.000001
    nmax = 1000
    iter = 0
    
    func = calc_cm(c_left,c_left,c_right)
    
    if (func < 0):
        x_a = c_left
    else:
        x_b = c_left

    func = calc_cm(c_right, c_left, c_right)
    if (func < 0):
        x_a = c_right
    else:
        x_b = c_right

    while (abs(x_a - x_b) > eps and iter < nmax):
        iter = iter + 1
        mid = (x_a + x_b) * 0.5

        func = calc_cm(mid, c_left, c_right)

        if (func < 0):
            x_a = mid
        else:
            x_b = mid

    c_mid = (x_a + x_b) * 0.5

    h_mid = c_mid * c_mid / g
    u_mid = 2 * (c_left - c_mid)
    
    v = h_mid * u_mid / (h_mid - h_right)

    if (x <= xdam - c_left * t):
        hex = h_left
    else:
        if (x <= xdam + (2 * c_left - 3 * c_mid) * t):
            hex = ( 4 / (9 * g) ) * ( c_left * c_left - c_left * (x - xdam) / t + (x - xdam) * (x - xdam) / (4 * t * t) )
        else:
            if (x <= xdam + v * t):
                hex = h_mid
            else:
                hex = h_right
    
    return hex

class Simulation1DDambreak:
    def __init__(
        self,
        solvers
    ):
        self.solvers      = solvers
        self.results      = {}
        self.epsilons     = [0, 1e-4, 1e-3, 1e-2]
        self.fields       = ["simtime", "runtime"]
        self.max_ref_lvls = [8, 9, 10, 11]
        
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
        
        # runs for verification
        for solver in self.solvers:
            for epsilon in self.epsilons:
                self.run(
                    solver=solver,
                    sim_time=2.5,
                    epsilon=epsilon,
                    L=8,
                    saveint=2.5,
                    limiter="on"
                )
                
                verification_depths = self.get_verification_depths()
                
                self.results["x"]                       = verification_depths["x"]
                self.results[solver][epsilon]["depths"] = verification_depths["depths"]
        
        # runs for speedups
        for solver in self.solvers:
            for epsilon in self.epsilons:
                for L in self.max_ref_lvls:
                    self.run(
                        solver=solver,
                        sim_time=40,
                        epsilon=epsilon,
                        L=L,
                        saveint=40,
                        limiter="off"
                    )
                    
                    results_dataframe = pd.read_csv( os.path.join("results", "cumulative-data.csv") )
                    
                    self.results[solver][epsilon][L]["simtime"] = results_dataframe["simtime"]
                    self.results[solver][epsilon][L]["runtime"] = results_dataframe["runtime"]
                    
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
        saveint,
        limiter
    ):
        print("Running simulation, L = " + str(L) + ", eps = " + str(epsilon) + ", solver: " + solver)
            
        test_script = os.path.join("..", "tests", "test.py")
        
        # using test.py to run simulations
        subprocess.run(
            [
                "python",
                test_script,       # test.py
                "run",             # ACTION
                solver,            # SOLVER
                "5",               # TEST_CASE
                str(sim_time),     # SIM_TIME
                str(epsilon),      # EPSILON
                str(L),            # MAX_REF_LVL
                str(saveint),      # SAVE_INT
                str(sim_time/100), # MASS_INT
                "surf",            # PLOT_TYPE
                limiter            # SLOPE_LIMITER
            ]
        )
        
    def plot_speedups(
        self,
        my_rc_params
    ):
        plt.rcParams.update(my_rc_params)
        
        fig, ax = plt.subplots()
        
        for solver in self.solvers:
            all_speedups = []
            
            # to find speedup axis limits
            for epsilon in self.epsilons:
                for L in self.max_ref_lvls:
                    interp_adaptive = scipy.interpolate.interp1d(
                        self.results[solver][epsilon][L]["simtime"],
                        self.results[solver][epsilon][L]["runtime"],
                        fill_value="extrapolate"
                    )
                    
                    interpolated_adaptive_runtime = interp_adaptive( self.results[solver][0][L]["simtime"] )
                    
                    all_speedups += (self.results[solver][0][L]["runtime"] / interpolated_adaptive_runtime).to_list()
                
            for epsilon in self.epsilons:
                if epsilon == 0: continue
                
                for L in self.max_ref_lvls:
                    if   L == 8:
                        color = "#F0C200"
                    elif L == 9:
                        color = "#FA9400"
                    elif L == 10:
                        color = "#E04000"
                    elif L == 11:
                        color = "#FA001A"
                    
                    interp_adaptive = scipy.interpolate.interp1d(
                        self.results[solver][epsilon][L]["simtime"],
                        self.results[solver][epsilon][L]["runtime"],
                        fill_value="extrapolate"
                    )
                    
                    interpolated_adaptive_runtime = interp_adaptive( self.results[solver][0][L]["simtime"] )
                    
                    ax.plot(
                        self.results[solver][0][L]["simtime"],
                        self.results[solver][0][L]["runtime"] / interpolated_adaptive_runtime,
<<<<<<< HEAD
                        label=(r"$L = %s$" % L) if np.isclose(epsilon, 1e-4) else None,
                        color=color
=======
                        linewidth=2,
                        label=r"$L = %s$" % L
>>>>>>> 957ee4d3ecd4a69573e95ac786653ee4a01f2ac4
                    )
                    
                xmin = self.results[solver][0][L]["simtime"].iloc[0]
                xmax = self.results[solver][0][L]["simtime"].iloc[-1]
                
                xlim = (xmin, xmax)
                
                ax.plot(
                    [xmin, xmax],
                    [1, 1],
                    linewidth=1,
                    linestyle="-.",
<<<<<<< HEAD
                    label="breakeven" if np.isclose(epsilon, 1e-4) else None,
=======
                    label="breakeven",
>>>>>>> 957ee4d3ecd4a69573e95ac786653ee4a01f2ac4
                    color='k'
                )
                
                ax.set_xlabel(r"$t \, (s)$")
                ax.set_ylabel( "Speedup ratio " + ("GPU-MWDG2/GPU-DG2" if solver == "mw" else "GPU-HWFV1/GPU-FV1") )
                ax.set_xlim(xlim)
                ax.set_ylim( 0, np.max(all_speedups) )
                ax.legend()    
                fig.savefig(os.path.join( "results", "runtimes-" + solver + "-eps-" + str(epsilon) ) + ".png", bbox_inches="tight")
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
                if epsilon == 0 or np.isclose(epsilon, 1e-3):
                    if epsilon == 0:
                        label = ("GPU-DG2"   if solver == "mw" else "GPU-FV1")
                        color = "#E50059"
                    elif np.isclose(epsilon, 1e-3):
                        label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-3}$"
                        color = "#7FED00"
                    
                    ax.plot(
                        self.results["x"],
                        self.results[solver][epsilon]["depths"],
                        label=label,
                        linewidth=2,
                        color=color
                    )
        
        ax.plot(
            self.results["x"],
            self.results[solver][0]["depths"],
            label="CPU-MWDG2" + r", $\epsilon = 10^{-3}$",
            linewidth=2,
            color="#FF9400"
        )
        
        t       = 2.5
        Lx      = 50
        xdam    = Lx/2
        h_left  = 6
        h_right = 2
        
        exact = [ calc_h_exact(xdam, x, Lx, h_left, h_right, t) for x in self.results["x"] ]
        
        ax.plot(
            self.results["x"],
            exact,
            label="Exact solution",
            color='k',
            linewidth=1
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
    Simulation1DDambreak( ["mw"] ).plot()