# gpu-mwdg2

This project is a shallow water model that is based on this paper. 

To use this model you need to have an NVIDIA GPU and the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). You also need to have [Python](https://www.python.org/downloads/) as well as the following packages installed.

- `pandas`
- `numpy`
- `matplotlib`
- `imageio`

## Building the model executable

The model is built using CMake in Visual Studio 2019: building on Linux using CMake has not been documented yet. To build the model, open the folder `gpu-mwdg2` in Visual Studio and go to toolbar at the top. In the toolbar, select either the `x64-Debug` or `x64-Release` option from the dropdown menu. After selecting an option, from the toolbar click `Build > Rebuild All`.

## Running simulations using the model

If the `x64-Debug` option was selected, the built executable will be located in `gpu-mwdg2\out\build\x64-Debug` (`gpu-mwdg2\out\build\x64-Release` if `x64-Release` was selected). Go to the appropriate folder and make another folder. This folder will hold all the files necessary to run the model. Inside this folder, make a `results` folder and a `.par` file.

### Description of parameter (`.par`) file needed to run simulations

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

Below is a `.par` file created to run simulations for the conical island test case (Section 3.2.1 in [Kesserwani and Sharifian (2020)](https://www.sciencedirect.com/science/article/pii/S0309170820303079)).

```
test_case   0
max_ref_lvl 10
min_dt      1
respath     .\results
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
In addition to this `.par` file, four other files are needed:

- `conical-island.start`: [raster file](https://support.geocue.com/ascii-raster-files-asc/) describing the initial depth
- `conical-island.start.Qx`: raster file describing the initial discharge in the x-direction
- `conical-island.dem`: DEM file describing the topography
- `conical-island.stage`: text file containing the coordinates of stages where hydrographs are recorded

Assuming that the `.par` file is called `conical-island.par` and is located inside a folder called `conical-island`, and that the model executable was built using the `x64-Release` option, there is the following folder tree:

```
gpu-mwdg2
|
| ...
|
- out
  | -- build
       |
       | 
       - x64-Release
         |
         | ...
         |
         - gpu-mwdg2.exe
         - conical-island
           |
           - results
             |
             | ...
             |
           - conical-island.start
           - conical-island.start.Qx
           - conical-island.dem
           - conical-island.stage
```


Click into the search bar at the top of the File Explorer, type in `cmd` and press enter to open a command line. In the command line, type in `..\gpu-mwdg2.exe <FILENAME>.par` and press enter to run the model. where <FILENAME> is the name of a `.par` file.
