import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import collections

def EXIT_HELP():
    help_message = ('Use this tool as:\n' + 'python run-simulations.py <SOLVER>, SOLVER={hwfv1|mwdg2} to select either the GPU-HWFV1 or GPU-MWDG2 solver, respectively.\n')
    
    sys.exit(help_message)
        
class ExperimentalDataMonai:
    def __init__(self):
        self.gauge_data = {}
        
        print('Reading experimental data...')
        
        experimental_dataframe = pd.read_csv('MonaiValley_WaveGages.txt', delimiter='\t')
        
        self.time                  = experimental_dataframe['Time (sec)']
        self.gauge_data['Point 1'] = experimental_dataframe['Gage 1 (cm)']
        self.gauge_data['Point 2'] = experimental_dataframe['Gage 2 (cm)']
        self.gauge_data['Point 3'] = experimental_dataframe['Gage 3 (cm)']

A4 = (8.3, 11.7)

class SimulationMonai:
    def __init__(
            self,
            solver
        ):
            print('Creating fields for simulation results...')
            
            self.solver      = solver
            self.epsilons    = [1e-3, 1e-4, 0]
            self.dirroots    = ['eps-1e-3', 'eps-1e-4', 'eps-0']
            self.input_file  = 'monai.par'
            self.points      = ['Point 1', 'Point 2', 'Point 3']
            
            red_dd = lambda: collections.defaultdict(red_dd)
            
            self.results = red_dd()
            
            self.write_par_file()
            
            simulation_runs = 1
            
            for epsilon, dirroot_base in zip(self.epsilons, self.dirroots):
                dfs = []
                for run in range(simulation_runs):
                    dirroot = dirroot_base + '-' + str(run)
                    
                    #self.run(epsilon, dirroot)
                    
                    dfs.append(pd.read_csv(os.path.join(dirroot, 'res.cumu'))[1:])
                    
                    if run > 0:
                        continue
                    
                    gauge_dataframe = pd.read_csv(
                        os.path.join(dirroot, 'res.stage'),
                        skiprows=9,
                        delimiter=' ',
                        header=None
                    )[1:]
                    
                    self.results[epsilon]['gauge_data']['Point 1'] = gauge_dataframe.iloc[:,1]
                    self.results[epsilon]['gauge_data']['Point 2'] = gauge_dataframe.iloc[:,2]
                    self.results[epsilon]['gauge_data']['Point 3'] = gauge_dataframe.iloc[:,3]
                    
                    self.results[epsilon]['map'] = np.loadtxt(fname=os.path.join(dirroot, 'res-1.elev'), skiprows=6)
                
                df_concat  = pd.concat(dfs) # concatenating along rows
                df_stacked = df_concat.groupby(df_concat.index) # some rows have same index, so we can use it to stack
                self.results[epsilon]['cumu'] = df_stacked.mean()
    
    def write_par_file(self):
        with open(self.input_file, 'w') as fp:
            params = (
                'monai\n' +
                f'{self.solver}\n' +
                'cuda\n' +
                'raster_out\n' +
                'cumulative\n' +
                'refine_wall\n' +
                'ref_thickness 32\n' +
                'max_ref_lvl   10\n' +
                'epsilon       0\n' +
                'wall_height   0.5\n' +
                'initial_tstep 1\n' +
                'fpfric        0.01\n' +
                'sim_time      22.5\n' +
                'massint       0.1\n' +
                'saveint       22.5\n' +
                'DEMfile       monai.dem\n' +
                'startfile     monai.start\n' +
                'bcifile       monai.bci\n' +
                'bdyfile       monai.bdy\n' +
                'stagefile     monai.stage\n'
            )
            
            fp.write(params)
    
    def run(
            self,
            epsilon,
            dirroot
        ):
            print('Running simulation, eps = ' + str(epsilon) + ', solver: ' + solver)
            
            executable = 'gpu-mwdg2.exe' if sys.platform == 'win32' else 'gpu-mwdg2'
            
            command_line_args = [
                os.path.join('..', executable),
                '-epsilon', str(epsilon),
                '-dirroot', str(dirroot),
                self.input_file
            ]
            
            subprocess.run(command_line_args)
        
    def compute_root_mean_squared_errors(self):
        RMSE = [
            np.sqrt( np.square( self.results[1e-3]['gauge_data']['Point 1'].to_numpy() - self.results[0]['gauge_data']['Point 1'].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[1e-4]['gauge_data']['Point 1'].to_numpy() - self.results[0]['gauge_data']['Point 1'].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[1e-3]['gauge_data']['Point 2'].to_numpy() - self.results[0]['gauge_data']['Point 2'].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[1e-4]['gauge_data']['Point 2'].to_numpy() - self.results[0]['gauge_data']['Point 2'].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[1e-3]['gauge_data']['Point 3'].to_numpy() - self.results[0]['gauge_data']['Point 3'].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[1e-4]['gauge_data']['Point 3'].to_numpy() - self.results[0]['gauge_data']['Point 3'].to_numpy() ).mean() ),
            np.sqrt( np.square( self.results[1e-3]['map'] - self.results[0]['map'] ).mean() ),
            np.sqrt( np.square( self.results[1e-4]['map'] - self.results[0]['map'] ).mean() )
        ]
        
        return pd.DataFrame(
            [[RMSE[0], RMSE[1]], [RMSE[2], RMSE[3]], [RMSE[4], RMSE[5]], [RMSE[6], RMSE[7]]],
            [f'Time series at {point}' for point in self.points] + ['Map'],
            ['\epsilon = 10-3', '\epsilon = 10-4']
        )
        
    def compute_correlation(self):
        corr = [
            np.corrcoef(x=self.results[1e-3]['gauge_data']['Point 1'].to_numpy(), y=self.results[0]['gauge_data']['Point 1'].to_numpy() )[0][1],
            np.corrcoef(x=self.results[1e-4]['gauge_data']['Point 1'].to_numpy(), y=self.results[0]['gauge_data']['Point 1'].to_numpy() )[0][1],
            np.corrcoef(x=self.results[1e-3]['gauge_data']['Point 2'].to_numpy(), y=self.results[0]['gauge_data']['Point 2'].to_numpy() )[0][1],
            np.corrcoef(x=self.results[1e-4]['gauge_data']['Point 2'].to_numpy(), y=self.results[0]['gauge_data']['Point 2'].to_numpy() )[0][1],
            np.corrcoef(x=self.results[1e-3]['gauge_data']['Point 3'].to_numpy(), y=self.results[0]['gauge_data']['Point 3'].to_numpy() )[0][1],
            np.corrcoef(x=self.results[1e-4]['gauge_data']['Point 3'].to_numpy(), y=self.results[0]['gauge_data']['Point 3'].to_numpy() )[0][1],
            np.corrcoef(x=self.results[1e-3]['map'].flatten(), y=self.results[0]['map'].flatten() )[0][1],
            np.corrcoef(x=self.results[1e-4]['map'].flatten(), y=self.results[0]['map'].flatten() )[0][1]
        ]
        
        return pd.DataFrame(
            [[corr[0], corr[1]], [corr[2], corr[3]], [corr[4], corr[5]], [corr[6], corr[7]]],
            [f'Time series at {point}' for point in self.points] + ['Map'],
            ['\epsilon = 10-3', '\epsilon = 10-4']
        )
    
    def write_table(self):
        RMSE = self.compute_root_mean_squared_errors()
        corr = self.compute_correlation()
        
        table = pd.concat([RMSE, corr], axis=1, keys=['RMSE', 'r'])
        
        table.to_csv('table.csv')
    
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
                self.results[epsilon]['cumu']['simtime'],
                self.results[epsilon]['gauge_data']['Point 1'] + 0.123591 - 0.13535,
                linewidth=2.5
            )[0]
            
            axs[1].plot(
                self.results[epsilon]['cumu']['simtime'],
                self.results[epsilon]['gauge_data']['Point 2'] + 0.132484 - 0.13535,
                linewidth=2.5
            )
            
            axs[2].plot(
                self.results[epsilon]['cumu']['simtime'],
                self.results[epsilon]['gauge_data']['Point 3'] + 0.130107 - 0.13535,
                linewidth=2.5
            )
            
            lines.append(line)
            
        line = axs[0].scatter(
            exp_data.time,
            exp_data.gauge_data['Point 1'] / 100, # convert from cm to m
            facecolor='None',
            edgecolor='black',
            s=5
        )
        
        lines.append(line)
        
        axs[1].scatter(
            exp_data.time,
            exp_data.gauge_data['Point 2'] / 100,
            facecolor='None',
            edgecolor='black',
            s=5
        )
        
        axs[2].scatter(
            exp_data.time,
            exp_data.gauge_data['Point 3'] / 100,
            facecolor='None',
            edgecolor='black',
            s=5
        )
        
        axs[2].set_xlabel('$t$ (s)')
        axs[2].set_xlim( (0,22.5) )
        
        axs[0].set_title('Point 1')
        axs[1].set_title('Point 2')
        axs[2].set_title('Point 3')
        
        for ax in axs:
            ax.set_ylim( (-0.02, 0.05) )
            ax.set_ylabel('$h + z$ (m)')
        
        main_labels = [
            ('GPU-MWDG2' if self.solver == 'mwdg2' else 'GPU-HWFV1') + ', $\epsilon = 10^{-3}$',
            ('GPU-MWDG2' if self.solver == 'mwdg2' else 'GPU-HWFV1') + ', $\epsilon = 10^{-4}$',
            'GPU-DG2'    if self.solver == 'mwdg2' else 'GPU-HWFV1',
            'Experimental'
        ]
        
        axs[0].legend(handles=lines, labels=main_labels, bbox_to_anchor=(1.0,2.0),ncol=2)
        
        fig.tight_layout()
        
        fig.savefig('predictions-' + self.solver, bbox_inches='tight')
            
    def plot_speedups(self):
        fig, axs = plt.subplots(
            nrows=3,
            ncols=3,
            figsize=(A4[0]-2, A4[1]-6),
            sharex=True
        )
          
        ax_wet_cells    = axs[0,0]; 
        ax_rel_dg2      = axs[0,1]; ax_rel_dg2.sharey(ax_wet_cells)
        ax_rel_mra      = axs[0,2]; ax_rel_mra.sharey(ax_wet_cells)
        ax_inst_speedup = axs[1,0]
        ax_dt           = axs[1,1]
        ax_num_tstep    = axs[1,2]
        ax_cumu_split   = axs[2,0]
        ax_total        = axs[2,1]
        ax_speedup      = axs[2,2]
        
        ax_wet_cells.set_title('$N_{wet}$ (%)',                      fontsize='small')
        ax_rel_dg2.set_title('$R_{DG2}$ (%)',                        fontsize='small')
        ax_rel_mra.set_title('$R_{MRA}$ (%)',                        fontsize='small')
        ax_inst_speedup.set_title('$S_{inst}$ (-)',                  fontsize='small')
        ax_dt.set_title('$\Delta t$ (s)',                            fontsize='small')
        ax_num_tstep.set_title('$N_{\Delta t}$ (-)',                 fontsize='small')
        ax_cumu_split.set_title('$C_{MRA}$ (dotted), $C_{DG2}$ (s)', fontsize='small')
        ax_total.set_title('$C_{tot}$ (s)',                          fontsize='small')
        ax_speedup.set_title('$S_{acc}$ (-)',                        fontsize='small')
        
        linewidth = 1
        lw = linewidth
        
        unif_cumu_df = self.results[0]['cumu']
        num_cells_unif = 784 * 486
        
        for epsilon in self.epsilons[:-1]: # skip eps = 0
            cumu_df = self.results[epsilon]['cumu']
        
            wet_cells    = cumu_df['num_wet_cells']    / unif_cumu_df['num_wet_cells']
            rel_dg2      = cumu_df['inst_time_solver'] / unif_cumu_df['inst_time_solver']
            rel_mra      = cumu_df['inst_time_mra']    / unif_cumu_df['inst_time_solver']
            inst_speedup = 1 / (rel_dg2 + rel_mra)
            speedup      = unif_cumu_df['cumu_time_solver'] / cumu_df['runtime_total']
            
            ax_wet_cells.plot   (cumu_df['simtime'], 100 * wet_cells,             linewidth=lw)
            ax_rel_dg2.plot     (cumu_df['simtime'], 100 * rel_dg2,               linewidth=lw)
            ax_rel_mra.plot     (cumu_df['simtime'], 100 * rel_mra,               linewidth=lw)
            ax_inst_speedup.plot(cumu_df['simtime'], inst_speedup,                linewidth=lw)
            ax_dt.plot          (cumu_df['simtime'], cumu_df['dt'],               linewidth=lw)
            ax_num_tstep.plot   (cumu_df['simtime'], cumu_df['num_timesteps'],    linewidth=lw)
            ax_cumu_split.plot  (cumu_df['simtime'], cumu_df['cumu_time_solver'], linewidth=lw)
            ax_cumu_split.scatter(cumu_df['simtime'].iloc[::8], cumu_df['cumu_time_mra'].iloc[::8], marker='x', s=4, color=ax_cumu_split.get_lines()[-1].get_color())
            ax_total.plot       (cumu_df['simtime'], cumu_df['runtime_total'],    linewidth=lw)
            ax_speedup.plot     (cumu_df['simtime'], speedup,                     linewidth=lw)
        
        ax_dt.plot       (unif_cumu_df['simtime'], unif_cumu_df['dt'],               linewidth=lw)
        ax_num_tstep.plot(unif_cumu_df['simtime'], unif_cumu_df['num_timesteps'],    linewidth=lw)
        ax_total.plot    (unif_cumu_df['simtime'], unif_cumu_df['cumu_time_solver'], linewidth=lw)
        
        num_yticks = 5
        num_xticks = 5
        
        for ax in axs[-1]:
            ax.xaxis.set_major_locator(ticker.MaxNLocator(num_xticks))
            ax.set_xlabel('$t$ (s)')
            ax.tick_params(axis='x', labelsize='small')
        
        for ax in axs.flat:
            ax.set_xlim((0, max(unif_cumu_df['simtime'])))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(num_yticks))
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2,3), useMathText=True)
            ax.yaxis.get_offset_text().set_fontsize('small')
            ax.tick_params(labelsize='small')
            ax.grid(True)
            
        ax_wet_cells.legend(handles=ax_dt.get_lines(), labels=["$\epsilon = 10^{-3}$", "$\epsilon = 10^{-4}$", "GPU-DG2"], fontsize='x-small', loc='upper left')
        
        fig.tight_layout()
        fig.savefig('speedups-monai-' + self.solver + '.png', bbox_inches='tight')
        fig.savefig('speedups-monai-' + self.solver + '.svg', bbox_inches='tight')
        
    def plot(
        self,   
        exp_data
    ):
        self.plot_speedups()
        self.plot_exp_data(exp_data)
        self.write_table()
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        EXIT_HELP()
        
    dummy, solver = sys.argv
    
    if solver != 'hwfv1' and solver != 'mwdg2':
        EXIT_HELP()
    
    #subprocess.run( ['python', 'stage.py' ] )
    #subprocess.run( ['python', 'inflow.py'] )
    #subprocess.run( ['python', 'raster.py'] )
    
    SimulationMonai(solver).plot( ExperimentalDataMonai() )