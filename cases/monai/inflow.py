import numpy as np
import matplotlib.pyplot as plt

def plot_impact_wave():
    print("Plotting impact wave...")
    
    inflow_timeseries = np.loadtxt(fname="inflow.txt", skiprows=1)
    
    fig, ax = plt.subplots( figsize=(4,1.5) )
    
    ax.plot(inflow_timeseries[:,0], inflow_timeseries[:,1])
    
    ax.set_title("Impact wave at left boundary")
    ax.set_xlabel("$t$ (s)")
    ax.set_ylabel("$h + z$ (m)")
    
    ax.set_xlim( (0,22.5) )
    ax.set_ylim( (-0.02,0.02) )
    
    fig.tight_layout()
    
    fig.savefig("impact-wave.svg", bbox_inches="tight")

def write_bdy_file():
    print("Preparing .bdy file...")
    
    inflow_timeseries = np.loadtxt(fname="inflow.txt", skiprows=1)
    
    with open("monai.bdy", 'w') as fp:
        fp.write("\nTEST1\n")
        
        timeseries_len = inflow_timeseries.shape[0]
        
        fp.write(str(timeseries_len) + " seconds\n")
        
        for entry in inflow_timeseries:
            fp.write(str( entry[1] ) + " " + str( entry[0] ) + "\n")
            
def write_bci_file():
    print("Preparing .bci file...")
    
    with open("monai.bci", 'w') as fp:
        fp.write("W 0 3.402 HVAR TEST1")

if __name__ == "__main__":
    write_bci_file()
    write_bdy_file()
    plot_impact_wave()