import os
import numpy as np
import matplotlib.pyplot as plt

def write_bci_file():
    with open("hilo.bci", 'w') as fp:
        fp.write("N 1700 7010 HVAR INFLOW")
        
def write_bdy_file():
    inflow_timeseries = np.loadtxt( fname=os.path.join("input-data", "se.dat") )
    
    # tsunami signal only
    inflow_timeseries[:,1] -= inflow_timeseries[0,1]
    
    # adjust by datum of 30 metres
    inflow_timeseries[:,1] += 30
    
    with open("hilo.bdy", 'w') as fp:
        fp.write("\nINFLOW\n")
        
        timeseries_len = inflow_timeseries.shape[0]
        
        fp.write(str(timeseries_len) + " minutes\n")
        
        timeshift = inflow_timeseries[0,0]
        
        for entry in inflow_timeseries:
            fp.write(str( entry[1] ) + " " + str(entry[0] - timeshift) + "\n")
            
def main():
    write_bci_file()
    write_bdy_file()
    
if __name__ == "__main__":
    main()