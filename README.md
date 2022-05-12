# gpu-mwdg2

This project is a shallow water model that is based on this paper. 

To use this model you need to have an NVIDIA GPU and the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). You also need to have [Python](https://www.python.org/downloads/) as well as the following packages installed.

- `pandas`
- `numpy`
- `matplotlib`
- `imageio`

## Building the model executable

### Building on Windows

1. Open the folder `gpu-mwdg2` in Visual Studio
2. Go to toolbar at the top and select either the `x64-Debug` or `x64-Release` option from the dropdown menu
3. From the toolbar click `Build > Rebuild All`
4. If the `x64-Debug` option was selected, the built executable may be located in `gpu-mwdg2\out\build\x64-Debug` (`gpu-mwdg2\out\build\x64-Release` if `x64-Release` was selected).

### Building on Linux

1. Navigate to the `gpu-mwdg2` directory
2. Run `cmake -S . -B build` in the command line
3. Run `cmake --build build`
4. The built executable will be located in `gpu-mwdg2/build`

## Running simulations using the model

In addition to the built executable, several other files are needed when running simulations using the model. To understand which files are needed, consider the following example.

### Running a simulation for an example test case

To run simulations for this example test case (Section 3.1.2 of [Kesserwani and Sharifian, 2020](https://www.sciencedirect.com/science/article/pii/S0309170820303079)), the following files are needed:

- `conical-island.par`: text file containing all the desired parameters for running a simulation
- `conical-island.start`: [raster file](https://support.geocue.com/ascii-raster-files-asc/) describing the initial depth
- `conical-island.start.Qx`: raster file describing the initial discharge in the x-direction
- `conical-island.dem`: raster file describing the topography, also known as a known a digital elevation model (DEM)
- `conical-island.stage`: text file containing the coordinates of the points where simulated depth time series is recorded

To run simulations of the example test case do the following steps:

1. Copy the model executable to `gpu-mwdg2\cases`
2. Navigate to `gpu-mwdg2\cases\conical-island`
3. Make a folder called `results`
4. Create a parameter file (`conical-island.par`, content shown below)
5. Write the stage file (`conical-island.stage`, content shown below)
6. Generate the raster files (`conical-island.dem`, `conical-island.start`, `conical-island.start.Qx`) by running `python raster.py` in a command prompt
7. Run `..\gpu-mwdg2.exe conical-island.par`in a command prompt to start running the simulation

Doing steps 1 to 6 will give the following folder tree:

```
-- gpu-mwdg2
 |
 ...
 |
 | -- cases
    |
    ...
    |
    | -- conical-island
        |
        | -- results
           |
           ...
           |
        | conical-island.par
        | conical-island.dem
        | conical-island.start
        | conical-island.start.Qx
        | conical-island.stage
```

#### Stage file

```
4
9.36  13.8
10.36 13.8
12.96 11.22
15.56 13.8
```

#### Parameter (`.par`) file 

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

### What is a parameter file?

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
