# gpu-mwdg2

This project is a shallow water model that is based on this paper. 

To use this model you need to have an NVIDIA GPU and the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). You also need to have [Python](https://www.python.org/downloads/).

## Building the model executable

The model is built using CMake in Visual Studio 2019: building on Linux using CMake has not been documented yet. To build the model, open the folder `gpu-mwdg2` in Visual Studio and go to toolbar at the top. In the toolbar, select either the `x64-Debug` or `x64-Release` option from the dropdown menu. After selecting an option, from the toolbar click `Build > Rebuild All`.

## Running the model

Depending on whether the `x64-Debug` or `x64-Release` option was selected, the built executable will be located in `gpu-mwdg2\out\build\x64-Debug` or `gpu-mwdg2\out\build\x64-Release` respectively. Go to the appropriate folder and make another folder. This folder will contain all the files necessary to run the model. Inside this folder make another folder named `results`, which will contain all of the simulation results.

To run a simulation using the executable, a `.par` must be created. This file contains all the parameters needed to run a simulation. The parameters and what function they serve are shown in the table below, and an example `.par` file is included after the table.


| Parameter   | Description |
| ------------|-------------|
| test_case 	| 0           |
| max_ref_lvl	| 9           |
| min_dt		| 1           |
| respath		| .\results   |
| epsilon		| 1e-3        |
| fpfric 		| 0.01        |
| rasterroot	| monai       |
| bcifile		| monai.bci   |
| bdyfile		| monai.bdy   |
| stagefile	| monai.stage |
| tol_h		| 1e-3        |
| tol_q		| 0           |
| tol_s		| 1e-9        |
| g			| 9.80665     |
| massint		| 1           |
| vtk			| on          |
| planar		| on          |
| row_major	| off         |
| c_prop		| off         |
| sim_time	| 22.5        |
| solver		| mw          |
| wall_height	| 0.5         |

```
test_case 	0
max_ref_lvl	9
min_dt		1
respath		.\results
epsilon		1e-3
fpfric 		0.01
rasterroot	monai
bcifile		monai.bci
bdyfile		monai.bdy
stagefile	monai.stage
tol_h		1e-3
tol_q		0
tol_s		1e-9
g		9.80665
massint		1
vtk		on
planar		on
row_major	off
c_prop		off
sim_time	22.5
solver		mw
wall_height	0.5
```

Click into the search bar at the top of the File Explorer, type in `cmd` and press enter to open a command line. In the command line, type in `..\gpu-mwdg2.exe <FILENAME>.par` and press enter to run the model. where <FILENAME> is the name of a `.par` file.