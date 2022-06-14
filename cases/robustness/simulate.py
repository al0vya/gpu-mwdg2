import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def EXIT_HELP():
    help_message = (
        "This tool is used to verify the robustness of a solver. Run using:\n" +
        "python run.py <SOLVER>, SOLVER={hw|mw}"
    )
    
    sys.exit(help_message)

def run_c_prop_tests():
    with open("tests.txt", 'w') as fp:
        fp.write("19")
        
    test_script = os.path.join("..", "tests", "test.py")
    
    subprocess.run(
        [
            "python",
            test_script, # test.py
            "test",      # ACTION
            solver,      # SOLVER
            "1e-3",      # EPSILON
            "8",         # MAX_REF_LVL
            "surf",      # PLOT_TYPE
            "off"        # SLOPE_LIMITER
        ]
    )
    
    subprocess.run( ["matlab", "-nosplash", "-nodesktop", "-r", "\"main; exit\""] )

class SimulationThreeConesDamBreak:
    def __init__(
        self,
        solver,
        epsilons
    ):
        self.solver    = solver
        self.epsilons  = epsilons
        self.intervals = ["0 s", "6 s", "12 s"]
        self.results   = {}
        
        for epsilon in self.epsilons:
            self.results[epsilon] = {}
            for interval in self.intervals:
                self.results[epsilon][interval] = {}
        
        # runs for speedups
        for epsilon in self.epsilons:
            self.run_dam_break(
                solver=solver,
                epsilon=epsilon
            )
            
            mesh_info = pd.read_csv( os.path.join("results", "mesh_info.csv") )
            
            mesh_dim = int(mesh_info.iloc[0][r"mesh_dim"])
            
            self.results[epsilon]["0 s"]  = np.loadtxt(fname=os.path.join("results", "depths-0.csv"), skiprows=1).reshape(mesh_dim, mesh_dim)
            self.results[epsilon]["6 s"]  = np.loadtxt(fname=os.path.join("results", "depths-1.csv"), skiprows=1).reshape(mesh_dim, mesh_dim)
            self.results[epsilon]["12 s"] = np.loadtxt(fname=os.path.join("results", "depths-2.csv"), skiprows=1).reshape(mesh_dim, mesh_dim)
        
        xsz = int(mesh_info.iloc[0][r"xsz"])
        ysz = int(mesh_info.iloc[0][r"ysz"])
        
        xmin = mesh_info.iloc[0]["xmin"]
        xmax = mesh_info.iloc[0]["xmax"]
        ymin = mesh_info.iloc[0]["ymin"]
        ymax = mesh_info.iloc[0]["ymax"]
        
        x_dim = xmax - xmin
        y_dim = ymax - ymin
        
        N_x = 1
        N_y = 1
        
        if (xsz >= ysz):
            N_x = xsz / ysz
        else:
            N_y = ysz / xsz
        
        self.N_x, self.N_y = N_x, N_y
        
        x = np.linspace(xmin, xmax, xsz)
        y = np.linspace(ymin, ymax, ysz)
        
        self.X, self.Y = np.meshgrid(x, y)
                
    def run_dam_break(
        self,
        solver,
        epsilon
    ):
        print( "Running simulation, solver: " + solver + ", epsilon = " + str(epsilon) )
        
        test_script = os.path.join("..", "tests", "test.py")
        
        # using test.py to run simulations
        subprocess.run(
            [
                "python",
                test_script,  # test.py
                "run",        # ACTION
                solver,       # SOLVER
                "20",         # TEST_CASE
                "12",         # SIM_TIME
                str(epsilon), # EPSILON
                "8",          # MAX_REF_LVL
                "6",          # SAVE_INT
                "12",         # MASS_INT
                "cont",       # PLOT_TYPE
                "off"         # SLOPE_LIMITER
            ]
        )

    def plot_depths(
        self,
        params
    ):
        #plt.rcParams.update(params)
        
        size = 100
        
        fig = plt.figure( figsize=(6.25,5.5) )
        
        gridspec = fig.add_gridspec(
            nrows=2,
            ncols=2,
            height_ratios=[20, 1],
            wspace=0.35,
            hspace=0.3
        )
        
        axs_title = gridspec.subplots()
        
        axs_title[0, 0].set_title(  "GPU-DG2"     if self.solver == "mw" else "GPU-FV1", fontsize=11)
        axs_title[0, 1].set_title( ("GPU-MWDG2, " if self.solver == "mw" else "GPU-FV1, ") + r"$ L = 8, \epsilon = 10^{-3}$", fontsize=11 )
        
        for ax in axs_title.flatten(): ax.axis("off")
        
        ax_uniform  = gridspec[0, 0].subgridspec(nrows=3, ncols=1, hspace=0.48).subplots(sharex=True)
        ax_adaptive = gridspec[0, 1].subgridspec(nrows=3, ncols=1, hspace=0.48).subplots(sharex=True)
        
        plt.setp(
            [ax_uniform, ax_adaptive],
            ylabel=(r"$y$" + " (m)"),
            xlabel=(r"$x$" + " (m)")
        )
        
        all_h = []
        
        for epsilon in self.epsilons:
            for interval in self.intervals:
                all_h += self.results[epsilon][interval].tolist()
        
        min_h = np.min(all_h)
        max_h = np.max(all_h)
        
        levels = 20
        
        dh = (max_h - min_h) / levels
        
        h_levels = [ min_h + dh * i for i in range(levels + 1) ]
        
        contourset_u0  =  ax_uniform[0].contourf(self.X, self.Y, self.results[0]["0 s"],     levels=levels) # for legend without negative depth
        contourset_u6  =  ax_uniform[1].contourf(self.X, self.Y, self.results[0]["6 s"],     levels=h_levels)
        contourset_u12 =  ax_uniform[2].contourf(self.X, self.Y, self.results[0]["12 s"],    levels=h_levels)
        contourset_a0  = ax_adaptive[0].contourf(self.X, self.Y, self.results[1e-3]["0 s"],  levels=h_levels)
        contourset_a6  = ax_adaptive[1].contourf(self.X, self.Y, self.results[1e-3]["6 s"],  levels=h_levels)
        contourset_a12 = ax_adaptive[2].contourf(self.X, self.Y, self.results[1e-3]["12 s"], levels=h_levels)
        
        # time stamps
        ax_uniform[0].set_title(r"$t$" + " = 0 s",  fontsize=10, x=1.165, y=-0.45)
        ax_uniform[1].set_title(r"$t$" + " = 6 s",  fontsize=10, x=1.165, y=-0.45)
        ax_uniform[2].set_title(r"$t$" + " = 12 s", fontsize=10, x=1.165, y=-0.45)
        
        # colorbar
        gs      = axs_title[0, 0].get_gridspec()
        ax_cbar = fig.add_subplot( gs[-1,:] )
        
        colorbar = fig.colorbar(
            contourset_u0,
            orientation="horizontal",
            label='m',
            cax=ax_cbar
        )
        
        plt.savefig(os.path.join("results", "three-humps-depth-contours"), bbox_inches="tight")
        plt.close()
    
    def plot_errors(
        self,
        params
    ):
        plt.rcParams.update(params)
        
        size = 100
        
        fig, axs = plt.subplots(
            nrows=7,
            ncols=1,
            gridspec_kw={ "height_ratios" : [size, 1, size, 1, size, 1, size/15] },
            figsize=(8,10)
        )
        
        plt.setp( axs, ylabel=(r"$y \, (m)$"), xlabel=(r"$x \, (m)$") )
        
        all_errors = []
        
        for interval in self.intervals:
            # absolute deviation percentage
            errors = (
                np.abs( self.results[0][interval] - self.results[1e-3][interval] )
            )
            
            all_errors += errors.tolist()
        
        min_error = np.min(all_errors)
        max_error = np.max(all_errors)
        
        levels = 20
        
        d_error = (max_error - min_error) / levels
        
        error_levels = [ min_error + d_error * i for i in range(levels + 1) ]
        
        err_0  = np.abs( self.results[0]["0 s"]  - self.results[1e-3]["0 s"]  )
        err_6  = np.abs( self.results[0]["6 s"]  - self.results[1e-3]["6 s"]  )
        err_12 = np.abs( self.results[0]["12 s"] - self.results[1e-3]["12 s"] )
        
        with open(os.path.join("results", "errors.txt"), 'w') as fp:
            fp.write( "0  s %s\n" % err_0.mean() )
            fp.write( "6  s %s\n" % err_6.mean() )
            fp.write( "12 s %s\n" % err_12.mean() )
        
        contourset_0  = axs[0].contourf(self.X, self.Y, err_0,  levels=error_levels) # for legend without negative depth
        contourset_6  = axs[2].contourf(self.X, self.Y, err_6,  levels=error_levels)
        contourset_12 = axs[4].contourf(self.X, self.Y, err_12, levels=error_levels)
        
        # get axis layout (subplots) of the figure
        gs = axs[0].get_gridspec()
        
        # remove bottom row subplots (axes)
        axs[1].remove()
        axs[3].remove()
        axs[5].remove()
        axs[6].remove()
        
        # add subplots for time stamps via set_title
        ax_0s   = fig.add_subplot(gs[1]); ax_0s.axis("off");  ax_0s.set_title(r"$t = 0 \, s$")
        ax_6s   = fig.add_subplot(gs[3]); ax_6s.axis("off");  ax_6s.set_title(r"$t = 6 \, s$")
        ax_12s  = fig.add_subplot(gs[5]); ax_12s.axis("off"); ax_12s.set_title(r"$t = 12 \, s$")
        
        # add a single subplot for the bottom row subplots
        ax_cbar = fig.add_subplot(gs[6]); 
        
        colorbar = fig.colorbar(contourset_12, cax=ax_cbar, orientation="horizontal")#, aspect=100)
        colorbar.ax.set_xlabel("Error")
        
        fig.tight_layout(h_pad=0)
        
        plt.savefig(os.path.join("results", "three-humps-error-contours.svg"), bbox_inches="tight")
        plt.close()
        
    def plot(self):
        my_rc_params = {
            "legend.fontsize" : "large",
            "axes.labelsize"  : "large",
            "axes.titlesize"  : "large",
            "xtick.labelsize" : "large",
            "ytick.labelsize" : "large"
        }
        
        self.plot_depths(my_rc_params)
        #self.plot_errors(my_rc_params)
        
if __name__ == "__main__":
    if len(sys.argv) < 2: EXIT_HELP()
    
    dummy, solver = sys.argv
    
    if solver != "hw" and solver != "mw": EXIT_HELP()
    
    #run_c_prop_tests()
    
    SimulationThreeConesDamBreak( solver, [0, 1e-3] ).plot()