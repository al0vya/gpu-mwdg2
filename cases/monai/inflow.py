import numpy as np

if __name__ == "__main__":
    print("Preparing boundary conditions (.bci file) and inflow timeseries (.bdy file)...")
    
    inflow_timeseries = np.loadtxt(fname="inflow.txt", skiprows=1)
    
    with open("monai.bci", 'w') as fp:
        fp.write("W 0 3.402 HVAR TEST1")
    
    with open("monai.bdy", 'w') as fp:
        fp.write("TEST1\n")
        
        timeseries_len = inflow_timeseries.shape[0]
        
        fp.write(str(timeseries_len) + " seconds\n")
        
        for entry in inflow_timeseries:
            fp.write(str( entry[1] ) + " " + str( entry[0] ) + "\n")