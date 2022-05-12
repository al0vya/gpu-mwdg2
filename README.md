# gpu-mwdg2

This project is a shallow water model that is based on this paper. 

To use this model you need to have an NVIDIA GPU and the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). You also need to have [Python](https://www.python.org/downloads/) as well as the following packages installed.

- `pandas`
- `numpy`
- `matplotlib`
- `imageio`

## Building the model executable

### Building on Windows

Open the folder `gpu-mwdg2` in Visual Studio and go to toolbar at the top. In the toolbar, select either the `x64-Debug` or `x64-Release` option from the dropdown menu. After selecting an option, from the toolbar click `Build > Rebuild All`. If the `x64-Debug` option was selected, the built executable may be located in `gpu-mwdg2\out\build\x64-Debug` (`gpu-mwdg2\out\build\x64-Release` if `x64-Release` was selected).

### Building on Linux

In the command line within the `gpu-mwdg2` directory, run `cmake -S . -B build` followed by `cmake --build build`. The built executable will be located in `gpu-mwdg2/build`.

## Running simulations using the model

To run simulations using the model, make a `results` directory and a `.par` file (described below).

### Description of parameter (`.par`) file

A `.par` file is a text file containing all of the parameters needed to run a simulation. The parameters and what function they serve are described in the table below. Many of the parameters are identical to [the ones](https://www.seamlesswave.com/Merewether1-1.html) used to run simulations using [LISFLOOD 8.0](https://www.seamlesswave.com/LISFLOOD8.0.html).

| Parameter   | Description |
| ------------|-------------|
| test_case 	| Enter 0 if running a real world test case, otherwise enter a number between 1 and 22 to run an in-built test case. |
| max_ref_lvl	| Maximum refinement level controlling the number of grids in the hierarchy. If running a real world test case, the finest grid's side length in terms of number of cells must be equal to or greater than the largest side length of the domain. |
| min_dt		| Minimum time step. |
| respath		| Relative path from the executable where simulation results are saved. |
| epsilon		| Error threshold epsilon. |
| fpfric 		| Manning's friction coefficient. |
| rasterroot	| For real world test cases. Root name of raster files e.g. the DEM data. |
| bcifile		| File specifying the locations and type of boundary cells. |
| bdyfile		| File containing time series data for time varying boundary conditions.   |
| stagefile	| File containing locations, in metre coordinates, where time series data for water depth should be saved. |
| tol_h		| Tolerance in metres for detecting dry cells; cells with depths below `tol_h` will be marked as dry. |
| tol_q		| N/A           |
| tol_s		| Speed threshold. |
| g			| Gravitational acceleration constant, `9.80665`.     |
| massint		| Interval in seconds at which `.vtk`, planar and row major data are saved. |
| vtk			| Flag controlling whether `vtk` data will be saved: can be either `on` or `off`. Flag is taken to be `off` if  not included in the `.par` file. |
| planar  | Flag to save data for making planar plots of the solution. |
| row_major	| Flag to save the solution data in row major format. |
| c_prop		| Flag to save discharge time series data (only for simulations with quiescent flow). |
| sim_time	| Simulation time. |
| solver		| Solver with which to run the simulation: `hw` for HWFV1 and `mw` for MWDG2. |
| wall_height	| Wall height for the computational domain in case the physical domain is not square. |

An example `.par` file and how it is used to run a simulation follows.

### Running a simulation for an example test case

Below is a `.par` file created to run simulations for the conical island test case (Section 3.2.1 in [Kesserwani and Sharifian, (2020)](https://www.sciencedirect.com/science/article/pii/S0309170820303079)).

```
test_case   0
max_ref_lvl 10
min_dt      1
respath     results
epsilon     1e-3
fpfric      0.01
rasterroot  conical-island
stagefile   conical-island.stage
tol_h       1e-3
tol_q       0
tol_s       1e-9
g           9.80665
massint     0.1
vtk         on
planar      on
row_major   off
c_prop      off
sim_time    22.5
solver      mw
wall_height 0.5
```
In addition to this `.par` file, four other files are needed for this example test case:

- `conical-island.start`: [raster file](https://support.geocue.com/ascii-raster-files-asc/) describing the initial depth
- `conical-island.start.Qx`: raster file describing the initial discharge in the x-direction
- `conical-island.dem`: DEM file describing the topography
- `conical-island.stage`: text file containing the coordinates of stages where simulated time series of the water depth is recorded

Copy the model executable to `gpu-mwdg2\cases`, start a command prompt at `gpu-mwdg2\cases\conical-island`, create a folder called `results` and finally run `..\gpu-mwdg2.exe conical-island.par` to start running the simulation. Before doing so however, the additional files (`conical-island.start`, `conical-island.stage`, etc) need to be created.

To create the raster files, in the command prompt run `python raster.py`, while the stage file (`conical-island.stage`) should read as:

```
4
9.36  13.8
10.36 13.8
12.96 11.22
15.56 13.8
```

These kinds of files are frequently needed to run simulations of real world test cases.

### Running automated simulations

In addition to creating files before running simulations (preprocessing), plotting results after running simulations (postprocessing) is also often desirable.

In the `gpu-mwdg2\cases` folder there are folders for a few commonly-used test cases with scripts that automatically preprocess files, run simulations and postprocess results.

For example, to run simulations for the conical island test case, simply run `python simulate.py` in the command. The `simulate.py` script will create all the necessary files for running the simulation, run the simulation and plot the results in `results` as `.png` files:

- `runtimes-hw`
- `runtimes-mw`
- `stage-#6`
- `stage-#9`
- `stage-#12`
- `stage-#22`

The results are plotted to reproduce the results in Section 3.2.1 of [Kesserwani and Sharifian, (2020)](https://www.sciencedirect.com/science/article/pii/S0309170820303079).
