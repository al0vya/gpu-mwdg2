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
        fp.write("19\n21\n22")
        
    test_script = os.path.join("..", "tests", "test.py")
    
    subprocess.run( ["python", test_script, "test", solver, "1e-3", "7", "surf"] )
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
        
        mesh_info = pd.read_csv( os.path.join("results", "mesh_info.csv") )
        
        mesh_dim = int(mesh_info.iloc[0][r"mesh_dim"])
        xsz      = int(mesh_info.iloc[0][r"xsz"])
        ysz      = int(mesh_info.iloc[0][r"ysz"])
        
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
        
        # runs for speedups
        for epsilon in self.epsilons:
            self.run(
                solver=solver,
                epsilon=epsilon
            )
            
            self.results[epsilon]["0 s"]  = np.loadtxt(fname=os.path.join("results", "depths-0.csv"), skiprows=1).reshape(mesh_dim, mesh_dim)
            self.results[epsilon]["6 s"]  = np.loadtxt(fname=os.path.join("results", "depths-1.csv"), skiprows=1).reshape(mesh_dim, mesh_dim)
            self.results[epsilon]["12 s"] = np.loadtxt(fname=os.path.join("results", "depths-2.csv"), skiprows=1).reshape(mesh_dim, mesh_dim)
                
    def run(
        self,
        solver,
        epsilon
    ):
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
                "cont"        # PLOT_TYPE
            ]
        )
    
    def plot(self):
        params = {
            "legend.fontsize" : "large",
            "axes.labelsize"  : "large",
            "axes.titlesize"  : "large",
            "xtick.labelsize" : "large",
            "ytick.labelsize" : "large"
        }
        
        plt.rcParams.update(params)
        
        size = 100
        
        fig, axs = plt.subplots(
            nrows=7,
            ncols=2,
            gridspec_kw={"height_ratios" : [size, 1, size, 1, size, 1, size/15]},
            figsize=(10,10)
        )
        
        plt.setp( axs, ylabel=(r"$y \, (m)$"), xlabel=(r"$x \, (m)$") )
        
        all_h = []
        
        for epsilon in self.epsilons:
            for interval in self.intervals:
                all_h += self.results[epsilon][interval].tolist()
        
        min_h = np.min(all_h)
        max_h = np.max(all_h)
        
        levels = 20
        
        dh = (max_h - min_h) / levels
        
        h_levels = [ min_h + dh * i for i in range(levels + 1) ]
        
        contourset_u0  = axs[0, 0].contourf(self.X, self.Y, self.results[0]["0 s"],     levels=levels) # for legend without negative depth
        contourset_u6  = axs[2, 0].contourf(self.X, self.Y, self.results[0]["6 s"],     levels=h_levels)
        contourset_u12 = axs[4, 0].contourf(self.X, self.Y, self.results[0]["12 s"],    levels=h_levels)
        contourset_a0  = axs[0, 1].contourf(self.X, self.Y, self.results[1e-3]["0 s"],  levels=h_levels)
        contourset_a6  = axs[2, 1].contourf(self.X, self.Y, self.results[1e-3]["6 s"],  levels=h_levels)
        contourset_a12 = axs[4, 1].contourf(self.X, self.Y, self.results[1e-3]["12 s"], levels=h_levels)
        
        axs[0, 0].set_title("GPU-DG2"     if solver == "mw" else "GPU-FV1")
        axs[0, 1].set_title( ("GPU-MWDG2, " if solver == "mw" else "GPU-HWFV1, ") + r"$L = 8, \, \epsilon = 10^{-3}$" )
        
        # get axis layout (subplots) of the figure
        gs = axs[0, 0].get_gridspec()
        
        # remove bottom row subplots (axes)
        for ax in axs[1, 0:]: ax.remove()
        for ax in axs[3, 0:]: ax.remove()
        for ax in axs[5, 0:]: ax.remove()
        for ax in axs[6, 0:]: ax.remove()
        
        # add subplots for time stamps via set_title
        ax_0s   = fig.add_subplot(gs[1, 0:]); ax_0s.axis("off");  ax_0s.set_title(r"$t = 0 \, s$")
        ax_6s   = fig.add_subplot(gs[3, 0:]); ax_6s.axis("off");  ax_6s.set_title(r"$t = 6 \, s$")
        ax_12s  = fig.add_subplot(gs[5, 0:]); ax_12s.axis("off"); ax_12s.set_title(r"$t = 12 \, s$")
        
        # add a single subplot for the bottom row subplots
        ax_cbar = fig.add_subplot(gs[6, 0:]); 
        
        colorbar = fig.colorbar(contourset_u0, cax=ax_cbar, orientation="horizontal")#, aspect=100)
        colorbar.ax.set_xlabel(r"$h \, (m)$")
        
        fig.tight_layout(h_pad=0)
        
        plt.savefig(os.path.join("results", "three-humps-depth-contours"), bbox_inches="tight")
        
if __name__ == "__main__":
    if len(sys.argv) < 2: EXIT_HELP()
    
    dummy, solver = sys.argv
    
    if solver != "hw" and solver != "mw": EXIT_HELP()
    
    run_c_prop_tests()
    
    #SimulationThreeConesDamBreak( solver, [0, 1e-3] ).plot()