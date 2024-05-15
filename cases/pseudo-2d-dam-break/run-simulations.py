import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.interpolate
import collections

def EXIT_HELP():
    help_message = ('Use this tool as:\n' + 'python run-simulations.py <SOLVER>, SOLVER={hwfv1|mwdg2} to select either the GPU-HWFV1 or GPU-MWDG2 solver, respectively.\n')
    
    sys.exit(help_message)

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
        h_ex = h_left
    else:
        if (x <= xdam + (2 * c_left - 3 * c_mid) * t):
            h_ex = ( 4 / (9 * g) ) * ( c_left * c_left - c_left * (x - xdam) / t + (x - xdam) * (x - xdam) / (4 * t * t) )
        else:
            if (x <= xdam + v * t):
                h_ex = h_mid
            else:
                h_ex = h_right
    
    return h_ex

class SimulationPseudo2DDambreak:
    def __init__(
        self,
        solver
    ):
        self.solver = solver
        self.epsilons = [1e-2, 1e-3, 1e-4, 0]
        self.dirroots = ['eps-1e-2', 'eps-1e-3', 'eps-1e-4', 'eps-0']
        self.input_file = 'inputs.par'
        self.max_ref_lvls = [8, 9, 10, 11][:-1]
        red_dd = lambda: collections.defaultdict(red_dd)
        self.results = red_dd()
        
        self.write_par_file()
        
        # runs for verification
        for epsilon, dirroot in zip(self.epsilons, self.dirroots):
            '''self.run(
                epsilon=epsilon,
                dirroot=dirroot + '-verify',
                sim_time=2.5,
                max_ref_lvl=8,
                tol_Krivo=10,
                saveint=2.5
            )'''
            
            self.results['x'], self.results[epsilon]['depths'] = self.get_depths(dirroot + '-verify')
            
        # runs for speedups
        for epsilon, dirroot in zip(self.epsilons, self.dirroots):
            for L in self.max_ref_lvls:
                '''self.run(
                    epsilon=epsilon,
                    dirroot=dirroot + f'-L-{L}',
                    sim_time=40,
                    max_ref_lvl=L,
                    tol_Krivo=9999,
                    saveint=9999
                )'''
                
                self.results[epsilon][L] = pd.read_csv( os.path.join(dirroot + f'-L-{L}', 'res.cumu') )[1:]
                    
    def write_par_file(self):
        with open(self.input_file, 'w') as fp:
            params = (
                f'{self.solver}\n' +
                'cuda\n' +
                'raster_out\n' +
                'cumulative\n' +
                'test_case     5\n' +
                'max_ref_lvl   8\n' +
                'epsilon       0\n' +
                'initial_tstep 1\n' +
                'sim_time      40\n' +
                'massint       0.1\n' +
                'saveint       40\n'
            )
            
            fp.write(params)
    
    def run(
            self,
            epsilon,
            dirroot,
            sim_time,
            max_ref_lvl,
            tol_Krivo,
            saveint
        ):
            print(f'Running simulation, eps = {epsilon}, L = {max_ref_lvl}, solver: {solver}')
            
            executable = 'gpu-mwdg2.exe' if sys.platform == 'win32' else 'gpu-mwdg2'
            
            command_line_args = [
                os.path.join('..', executable),
                '-epsilon', str(epsilon),
                '-dirroot', str(dirroot),
                '-sim_time', str(sim_time),
                '-max_ref_lvl', str(max_ref_lvl),
                *(['-tol_Krivo', str(tol_Krivo)] if tol_Krivo != 9999 else ['']),
                '-saveint', str(saveint),
                self.input_file
            ]
            
            subprocess.run(command_line_args)
            
    def get_depths(
        self,
        dirroot
    ):
        depths_raster = np.loadtxt(os.path.join(dirroot, 'res-1.wd'), skiprows=6)
        
        mesh_dim = depths_raster.shape[1]
        centre = int(mesh_dim / 2)
        xmin = 0
        xmax = 50
        dx = (xmax - xmin) / mesh_dim
        x = [ xmin + i * dx for i in range(mesh_dim) ]
        
        x[-1] += dx
        
        depths = depths_raster[centre]
        
        return x, depths
        
    def plot_speedups(
        self
    ):
        A4 = (8.3, 11.7)
        fig = plt.figure( figsize=(A4[0]-2, A4[1]-3) )
        
        gridspec = fig.add_gridspec(
            ncols=len(self.max_ref_lvls), # one column per L
            nrows=10,
            hspace=0.4
        )
        
        axs = gridspec.subplots(sharex=True)#, sharey='row')
        
        if axs.ndim == 1:
            axs = axs.reshape((10,1))
        
        for i, ax in enumerate(axs[0]):
            ax.set_title(f'$L = {8+i}$')
        
        axs.T[0,0].set_ylabel('A (%)')
        axs.T[0,1].set_ylabel('$R_{FV1}$ (%)')
        axs.T[0,2].set_ylabel('$I_{FV1}$ (ms)')
        axs.T[0,3].set_ylabel('$I_{MRA}$ (ms)')
        axs.T[0,4].set_ylabel('$\Delta t$')
        axs.T[0,5].set_ylabel('$N_{\Delta t}$')
        axs.T[0,6].set_ylabel('$C_{MRA}$ (s)')
        axs.T[0,7].set_ylabel('$C_{FV1}$ (s)')
        axs.T[0,8].set_ylabel('$C_{tot}$ (s)')
        axs.T[0,9].set_ylabel('Speedup')
        
        for ax in axs[:2,:].flat:
            ax.sharey(axs[0,0])
            
        for axs_ in axs[2:,:]:
            for ax in axs_:
                ax.sharey(axs_[0])
            
        linewidth = 1
        lw = linewidth
        
        for axs_v, L in zip(axs.T, self.max_ref_lvls):
            for i, epsilon in enumerate(self.epsilons[:-1]):
                total_interp = scipy.interpolate.interp1d(
                    self.results[epsilon][L]['simtime'],
                    self.results[epsilon][L]['runtime_total'],
                    fill_value='extrapolate'
                )
                
                num_cells_interp = scipy.interpolate.interp1d(
                    self.results[epsilon][L]['simtime'],
                    self.results[epsilon][L]['num_cells'],
                    fill_value='extrapolate'
                )
                
                inst_time_solver_interp = scipy.interpolate.interp1d(
                    self.results[epsilon][L]['simtime'],
                    self.results[epsilon][L]['inst_time_solver'],
                    fill_value='extrapolate'
                )
                
                time = self.results[0][L]['simtime']
                num_timesteps_uniform = self.results[0][L]['num_timesteps']
                runtime_adaptive = total_interp(time)
                runtime_uniform = self.results[0][L]['runtime_total']
                speedup = runtime_uniform / runtime_adaptive
                
                axs_v[0].plot(time, 100 * num_cells_interp(time) / self.results[0][L]['num_cells'],        linewidth=lw, label=f'$L$ = {L}')
                axs_v[1].plot(time, 100 * inst_time_solver_interp(time) / self.results[0][L]['inst_time_solver'],        linewidth=lw, label=f'$L$ = {L}')
                axs_v[2].plot(self.results[epsilon][L]['simtime'], 1000 * self.results[epsilon][L]['inst_time_solver'], linewidth=lw, label=f'$L$ = {L}')
                axs_v[3].plot(self.results[epsilon][L]['simtime'], 1000 * self.results[epsilon][L]['inst_time_mra'],    linewidth=lw, label=f'$L$ = {L}')
                axs_v[4].plot(self.results[epsilon][L]['simtime'], self.results[epsilon][L]['dt'],               linewidth=lw, label=f'$L$ = {L}')
                axs_v[5].plot(self.results[epsilon][L]['simtime'], self.results[epsilon][L]['num_timesteps'],    linewidth=lw, label=f'$L$ = {L}')
                axs_v[6].plot(self.results[epsilon][L]['simtime'], self.results[epsilon][L]['cumu_time_mra'],    linewidth=lw, label=f'$L$ = {L}')
                axs_v[7].plot(self.results[epsilon][L]['simtime'], self.results[epsilon][L]['cumu_time_solver'], linewidth=lw, label=f'$L$ = {L}')
                axs_v[8].plot(self.results[epsilon][L]['simtime'], self.results[epsilon][L]['runtime_total'],    linewidth=lw, label=f'$L$ = {L}')
                axs_v[9].plot(time, speedup,         linewidth=lw, label=f'$L$ = {L}')
        
        slices_num_timesteps = [(self.results[0][L]['num_timesteps']    < np.max(self.results[1e-4][self.max_ref_lvls[-1]]['num_timesteps'])).values for L in self.max_ref_lvls]
        slices_runtime_total = [(self.results[0][L]['cumu_time_solver'] < np.max(self.results[1e-4][self.max_ref_lvls[-1]]['runtime_total'])).values for L in self.max_ref_lvls]
        
        for ax, L, slice in zip(axs[5], self.max_ref_lvls, slices_num_timesteps):
            ax.plot(self.results[0][L]['simtime'].iloc[slice], self.results[0][L]['num_timesteps'].iloc[slice], linewidth=lw, color='black', linestyle='--')
        
        for ax, L, slice in zip(axs[8], self.max_ref_lvls, slices_runtime_total):
            ax.plot(self.results[0][L]['simtime'].iloc[slice], self.results[0][L]['cumu_time_solver'].iloc[slice], linewidth=lw, color='black', linestyle='--')
        
        num_yticks = 5
        num_xticks = 5
        
        for ax in axs.T[0]:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(num_yticks))
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,3))
            ax.yaxis.get_offset_text().set_fontsize('small')
            ax.tick_params(axis='y', labelsize='small')
        
        for ax in axs[-1]:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(num_xticks))
            ax.set_xlabel('$t$ (s)')
            ax.tick_params(axis='x', labelsize='small')
        
        for ax in axs[:,1:].flat:
            ax.axes.tick_params(axis='y', labelleft=False)
        
        for ax in axs.flat:
            ax.grid(True)
        
        #fig.tight_layout()        
        fig.savefig(os.path.join( 'res', 'runtimes-' + solver) + '.png', bbox_inches='tight')
        fig.savefig(os.path.join( 'res', 'runtimes-' + solver) + '.svg', bbox_inches='tight')
        
    def plot_depths(
        self
    ):
        fig, ax = plt.subplots( figsize=(2.75, 2.5) )
        
        for epsilon in self.epsilons:
            if epsilon == 0:
                label = ('GPU-DG2'   if solver == 'mwdg2' else 'GPU-FV1')
            elif np.isclose(epsilon, 1e-2):
                label = ('GPU-MWDG2' if solver == 'mwdg2' else 'GPU-HWFV1') + r', $\epsilon = 10^{-2}$'
            elif np.isclose(epsilon, 1e-3):
                label = ('GPU-MWDG2' if solver == 'mwdg2' else 'GPU-HWFV1') + r', $\epsilon = 10^{-3}$'
            elif np.isclose(epsilon, 1e-4):
                label = ('GPU-MWDG2' if solver == 'mwdg2' else 'GPU-HWFV1') + r', $\epsilon = 10^{-4}$'
            
            if epsilon == 0 or np.isclose(epsilon, 1e-3):
                ax.plot(
                    self.results['x'],
                    self.results[epsilon]['depths'],
                    label=label
                )
            
        t       = 2.5
        Lx      = 50
        xdam    = Lx/2
        h_left  = 6
        h_right = 2
        
        exact = [ calc_h_exact(xdam, x, Lx, h_left, h_right, t) for x in self.results['x'] ]
        
        ax.plot(
            self.results['x'],
            exact,
            label='Exact solution',
            color='k',
            linewidth=1
        )
        
        xlim = ( self.results['x'][0], self.results['x'][-1] )
        
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$h$ (m)')
        #ax.set_xlim(xlim)
        ax.legend()
        fig.savefig(os.path.join('res', 'verification.png'), bbox_inches='tight')
        
        plt.close()
        
    def plot(self):
        self.plot_speedups()
        self.plot_depths()
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        EXIT_HELP()
        
    dummy, solver = sys.argv
    
    if solver != 'hwfv1' and solver != 'mwdg2':
        EXIT_HELP()
    
    SimulationPseudo2DDambreak(solver).plot()