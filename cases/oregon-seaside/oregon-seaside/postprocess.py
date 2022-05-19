import os
import numpy as np
import matplotlib.pyplot as plt

def load_experimental_gauge_timeseries():
    experimental_gauges = {}
    
    experimental_gauges["W1"] = {}
    experimental_gauges["W2"] = {}
    experimental_gauges["W3"] = {}
    experimental_gauges["W4"] = {}
    experimental_gauges["A1"] = {}
    experimental_gauges["A2"] = {}
    experimental_gauges["A3"] = {}
    experimental_gauges["A4"] = {}
    experimental_gauges["A5"] = {}
    experimental_gauges["A6"] = {}
    experimental_gauges["A7"] = {}
    experimental_gauges["A8"] = {}
    experimental_gauges["A9"] = {}
    experimental_gauges["B1"] = {}
    experimental_gauges["B2"] = {}
    experimental_gauges["B3"] = {}
    experimental_gauges["B4"] = {}
    experimental_gauges["B5"] = {}
    experimental_gauges["B6"] = {}
    experimental_gauges["B7"] = {}
    experimental_gauges["B8"] = {}
    experimental_gauges["B9"] = {}
    experimental_gauges["C1"] = {}
    experimental_gauges["C2"] = {}
    experimental_gauges["C3"] = {}
    experimental_gauges["C4"] = {}
    experimental_gauges["C5"] = {}
    experimental_gauges["C6"] = {}
    experimental_gauges["C7"] = {}
    experimental_gauges["C8"] = {}
    experimental_gauges["C9"] = {}
    experimental_gauges["D1"] = {}
    experimental_gauges["D2"] = {}
    experimental_gauges["D3"] = {}
    experimental_gauges["D4"] = {}
    
    wavegage = np.loadtxt(fname=os.path.join("comparison-data", "Wavegage.txt"), skiprows=1)
    
    A1 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A1.txt"), skiprows=3)
    A2 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A2.txt"), skiprows=3)
    A3 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A3.txt"), skiprows=3)
    A4 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A4.txt"), skiprows=3)
    A5 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A5.txt"), skiprows=3)
    A6 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A6.txt"), skiprows=3)
    A7 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A7.txt"), skiprows=3)
    A8 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A8.txt"), skiprows=3)
    A9 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A9.txt"), skiprows=3)
    B1 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B1.txt"), skiprows=3)
    B2 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B2.txt"), skiprows=3)
    B3 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B3.txt"), skiprows=3)
    B4 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B4.txt"), skiprows=3)
    B5 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B5.txt"), skiprows=3)
    B6 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B6.txt"), skiprows=3)
    B7 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B7.txt"), skiprows=3)
    B8 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B8.txt"), skiprows=3)
    B9 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B9.txt"), skiprows=3)
    C1 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C1.txt"), skiprows=3)
    C2 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C2.txt"), skiprows=3)
    C3 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C3.txt"), skiprows=3)
    C4 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C4.txt"), skiprows=3)
    C5 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C5.txt"), skiprows=3)
    C6 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C6.txt"), skiprows=3)
    C7 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C7.txt"), skiprows=3)
    C8 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C8.txt"), skiprows=3)
    C9 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C9.txt"), skiprows=3)
    D1 = np.loadtxt(fname=os.path.join("comparison-data", "Location_D1.txt"), skiprows=3)
    D2 = np.loadtxt(fname=os.path.join("comparison-data", "Location_D2.txt"), skiprows=3)
    D3 = np.loadtxt(fname=os.path.join("comparison-data", "Location_D3.txt"), skiprows=3)
    D4 = np.loadtxt(fname=os.path.join("comparison-data", "Location_D4.txt"), skiprows=3)
    
    experimental_gauges["W1"]["time"]    = wavegage[:,0]
    experimental_gauges["W2"]["time"]    = wavegage[:,0]
    experimental_gauges["W3"]["time"]    = wavegage[:,0]
    experimental_gauges["W4"]["time"]    = wavegage[:,0]
    
    experimental_gauges["W1"]["history"] = wavegage[:,3]
    experimental_gauges["W2"]["history"] = wavegage[:,4]
    experimental_gauges["W3"]["history"] = wavegage[:,5]
    experimental_gauges["W4"]["history"] = wavegage[:,6]
    
    experimental_gauges["A1"]["time"] = A1[:,0]
    experimental_gauges["A2"]["time"] = A2[:,0]
    experimental_gauges["A3"]["time"] = A3[:,0]
    experimental_gauges["A4"]["time"] = A4[:,0]
    experimental_gauges["A5"]["time"] = A5[:,0]
    experimental_gauges["A6"]["time"] = A6[:,0]
    experimental_gauges["A7"]["time"] = A7[:,0]
    experimental_gauges["A8"]["time"] = A8[:,0]
    experimental_gauges["A9"]["time"] = A9[:,0]
    experimental_gauges["B1"]["time"] = B1[:,0]
    experimental_gauges["B2"]["time"] = B2[:,0]
    experimental_gauges["B3"]["time"] = B3[:,0]
    experimental_gauges["B4"]["time"] = B4[:,0]
    experimental_gauges["B5"]["time"] = B5[:,0]
    experimental_gauges["B6"]["time"] = B6[:,0]
    experimental_gauges["B7"]["time"] = B7[:,0]
    experimental_gauges["B8"]["time"] = B8[:,0]
    experimental_gauges["B9"]["time"] = B9[:,0]
    experimental_gauges["C1"]["time"] = C1[:,0]
    experimental_gauges["C2"]["time"] = C2[:,0]
    experimental_gauges["C3"]["time"] = C3[:,0]
    experimental_gauges["C4"]["time"] = C4[:,0]
    experimental_gauges["C5"]["time"] = C5[:,0]
    experimental_gauges["C6"]["time"] = C6[:,0]
    experimental_gauges["C7"]["time"] = C7[:,0]
    experimental_gauges["C8"]["time"] = C8[:,0]
    experimental_gauges["C9"]["time"] = C9[:,0]
    experimental_gauges["D1"]["time"] = D1[:,0]
    experimental_gauges["D2"]["time"] = D2[:,0]
    experimental_gauges["D3"]["time"] = D3[:,0]
    experimental_gauges["D4"]["time"] = D4[:,0]
    
    experimental_gauges["A1"]["history"] = A1[:,1]
    experimental_gauges["A2"]["history"] = A2[:,1]
    experimental_gauges["A3"]["history"] = A3[:,1]
    experimental_gauges["A4"]["history"] = A4[:,1]
    experimental_gauges["A5"]["history"] = A5[:,1]
    experimental_gauges["A6"]["history"] = A6[:,1]
    experimental_gauges["A7"]["history"] = A7[:,1]
    experimental_gauges["A8"]["history"] = A8[:,1]
    experimental_gauges["A9"]["history"] = A9[:,1]
    experimental_gauges["B1"]["history"] = B1[:,1]
    experimental_gauges["B2"]["history"] = B2[:,1]
    experimental_gauges["B3"]["history"] = B3[:,1]
    experimental_gauges["B4"]["history"] = B4[:,1]
    experimental_gauges["B5"]["history"] = B5[:,1]
    experimental_gauges["B6"]["history"] = B6[:,1]
    experimental_gauges["B7"]["history"] = B7[:,1]
    experimental_gauges["B8"]["history"] = B8[:,1]
    experimental_gauges["B9"]["history"] = B9[:,1]
    experimental_gauges["C1"]["history"] = C1[:,1]
    experimental_gauges["C2"]["history"] = C2[:,1]
    experimental_gauges["C3"]["history"] = C3[:,1]
    experimental_gauges["C4"]["history"] = C4[:,1]
    experimental_gauges["C5"]["history"] = C5[:,1]
    experimental_gauges["C6"]["history"] = C6[:,1]
    experimental_gauges["C7"]["history"] = C7[:,1]
    experimental_gauges["C8"]["history"] = C8[:,1]
    experimental_gauges["C9"]["history"] = C9[:,1]
    experimental_gauges["D1"]["history"] = D1[:,1]
    experimental_gauges["D2"]["history"] = D2[:,1]
    experimental_gauges["D3"]["history"] = D3[:,1]
    experimental_gauges["D4"]["history"] = D4[:,1]
    
    return experimental_gauges

def load_computed_gauge_timeseries(
    stagefile
):
    print("Loading computed gauges timeseries: %s..." % stagefile)
    
    gauges = np.loadtxt(stagefile, skiprows=42, delimiter=" ")
    
    datum = -0.00202286243437291
    
    return {
        "time" : gauges[:,0],
        "BD"   : gauges[:,1],
        "W1"   : gauges[:,2],
        "W2"   : gauges[:,3],
        "W3"   : gauges[:,4],
        "W4"   : gauges[:,5],
        "A1"   : gauges[:,6],
        "A2"   : gauges[:,7],
        "A3"   : gauges[:,8],
        "A4"   : gauges[:,9],
        "A5"   : gauges[:,10],
        "A6"   : gauges[:,11],
        "A7"   : gauges[:,12],
        "A8"   : gauges[:,13],
        "A9"   : gauges[:,14],
        "B1"   : gauges[:,15],
        "B2"   : gauges[:,16],
        "B3"   : gauges[:,17],
        "B4"   : gauges[:,18],
        "B5"   : gauges[:,19],
        "B6"   : gauges[:,20],
        "B7"   : gauges[:,21],
        "B8"   : gauges[:,22],
        "B9"   : gauges[:,23],
        "C1"   : gauges[:,24],
        "C2"   : gauges[:,25],
        "C3"   : gauges[:,26],
        "C4"   : gauges[:,27],
        "C5"   : gauges[:,28],
        "C6"   : gauges[:,29],
        "C7"   : gauges[:,30],
        "C8"   : gauges[:,31],
        "C9"   : gauges[:,32],
        "D1"   : gauges[:,33],
        "D2"   : gauges[:,34],
        "D3"   : gauges[:,35],
        "D4"   : gauges[:,36],
    }
    
def read_stage_elevations(
    stagefile
):
    header = []
    
    with open(stagefile, 'r') as fp:
        for i, line in enumerate(fp):
            if i > 2:
                header.append(line)
                
            if i > 38:
                break
    
    return {
        "BD" : float( header[0 ].split()[3] ),
        "W1" : float( header[1 ].split()[3] ),
        "W2" : float( header[2 ].split()[3] ),
        "W3" : float( header[3 ].split()[3] ),
        "W4" : float( header[4 ].split()[3] ),
        "A1" : float( header[5 ].split()[3] ),
        "A2" : float( header[6 ].split()[3] ),
        "A3" : float( header[7 ].split()[3] ),
        "A4" : float( header[8 ].split()[3] ),
        "A5" : float( header[9 ].split()[3] ),
        "A6" : float( header[10].split()[3] ),
        "A7" : float( header[11].split()[3] ),
        "A8" : float( header[12].split()[3] ),
        "A9" : float( header[13].split()[3] ),
        "B1" : float( header[14].split()[3] ),
        "B2" : float( header[15].split()[3] ),
        "B3" : float( header[16].split()[3] ),
        "B4" : float( header[17].split()[3] ),
        "B5" : float( header[18].split()[3] ),
        "B6" : float( header[19].split()[3] ),
        "B7" : float( header[20].split()[3] ),
        "B8" : float( header[21].split()[3] ),
        "B9" : float( header[22].split()[3] ),
        "C1" : float( header[23].split()[3] ),
        "C2" : float( header[24].split()[3] ),
        "C3" : float( header[25].split()[3] ),
        "C4" : float( header[26].split()[3] ),
        "C5" : float( header[27].split()[3] ),
        "C6" : float( header[28].split()[3] ),
        "C7" : float( header[29].split()[3] ),
        "C8" : float( header[30].split()[3] ),
        "C9" : float( header[31].split()[3] ),
        "D1" : float( header[32].split()[3] ),
        "D2" : float( header[33].split()[3] ),
        "D3" : float( header[34].split()[3] ),
        "D4" : float( header[35].split()[3] ),
    }

def compare_timeseries_stage(
    stagefiles,
    experimental_gauges,
    name
):
    print("Comparing timeseries at gauge %s..." % name)
    
    my_rc_params = {
        "legend.fontsize" : "large",
        "axes.labelsize"  : "large",
        "axes.titlesize"  : "large",
        "xtick.labelsize" : "large",
        "ytick.labelsize" : "large"
    }
    
    plt.rcParams.update(my_rc_params)
    
    fig, ax = plt.subplots()
    
    for stagefile in stagefiles:
        elevations      = read_stage_elevations(stagefile)
        computed_gauges = load_computed_gauge_timeseries(stagefile)
        
        ax.plot(
            computed_gauges["time"],
            computed_gauges[name],
            label=stagefile
        )
    
    ax.scatter(
        experimental_gauges[name]["time"][::8],
        experimental_gauges[name]["history"][::8],
        label="experimental",
        edgecolor="#000000",
        facecolor="None"
    )
    
    ylim = (
        np.min( experimental_gauges[name]["history"] ),
        np.max( experimental_gauges[name]["history"] )
    )
    
    plt.setp(
        ax,
        title=name,
        xlim=(0,40),
        #ylim=ylim,
        xlabel=r"$t \, (hr)$",
        ylabel=r"$h + z \, (m)$"
    )
    
    plt.legend()
    
    fig.savefig(name, bbox_inches="tight")
    
    plt.close()
    
def main():
    stagefiles = [
        "stage-mwdg2-1e-3.wd",
        "stage-mwdg2-1e-4.wd"
    ]
     
    experimental_gauges = load_experimental_gauge_timeseries()
    
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="W1")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="W2")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="W3")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="W4")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A1")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A2")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A3")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A4")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A5")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A6")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A7")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A8")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A9")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B1")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B2")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B3")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B4")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B5")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B6")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B7")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B8")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B9")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C1")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C2")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C3")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C4")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C5")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C6")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C7")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C8")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C9")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="D1")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="D2")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="D3")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="D4")
        
if __name__ == "__main__":
    main()