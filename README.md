# gpu-mwdg2

This project is a 2D shallow water model based on [this paper](https://iwaponline.com/jh/article/doi/10.2166/hydro.2023.154/95732/GPU-parallelisation-of-wavelet-based-grid) for running real-world simulations. 

To run simulations using the model, you need to have an NVIDIA GPU and the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) installed. You also need to have [Python](https://www.python.org/downloads/) and the following packages installed:

- `pandas`
- `numpy`
- `matplotlib`
- `imageio`

## 1 Running simulations using the model: building the executable

### 1.1 Building on Windows

1. Open the folder `gpu-mwdg2` in Visual Studio
2. Go to toolbar at the top and select either the `x64-Debug` or `x64-Release` option from the dropdown menu
3. From the toolbar click `Build > Rebuild All`
4. If the `x64-Debug` option was selected, the built executable may be located in `gpu-mwdg2\out\build\x64-Debug` (`gpu-mwdg2\out\build\x64-Release` if `x64-Release` was selected).

### 1.2 Building on Linux*

1. Navigate to the `gpu-mwdg2` directory
2. Run `cmake -S . -B build` in the command line
3. Run `cmake --build build`
4. The built executable will be located in `gpu-mwdg2/build`

* There is a potential issue about building on Linux because the [CUB library](https://nvlabs.github.io/cub/) is not properly detected. This issue be avoided by changing all `#include`s with `cub/` to `../../cub/`.

## 2 Running simulations using the model: preparing the input files

Running simulations using the model follows a very similar workflow as [LISFLOOD-FP](http://www.bristol.ac.uk/geography/research/hydrology/models/lisflood/). A summary of the workflow is provided in this readme, whereas [more detailed guidance is available on the SEAMLESS-WAVE website](https://www.seamlesswave.com/LISFLOOD8.0.html). The workflow requires preparing several input files, as listed in the table below.

|Input file|File extension|Description|
|----------|--------------|-----------|
|Digital elevation model|`.dem`|ASCII [raster file](https://support.geocue.com/ascii-raster-files-asc/) containing the numerical values of the bathymetric elevation pixel-by-pixel.|
|Initial flow conditions|`.start,.start.Qx,.start.Qy`|ASCII raster file containing the numerical values of the initial water depth and discharge pixel-by-pixel.|
|Boundary conditions|`.bci`|Text file specifying where boundary conditions are enforced and what type (fixed versus time-varying).|
|Point source locations|`.bci`|Text file specifying the locations of point sources and what type (fixed versus time-varying).|
|Time series at boundaries and point sources|`.bdy`|Text file containing time series in case time-varying boundary conditions and/or point sources have been specified in the .bci file.|
|Stage and gauge locations|`.stage,.guage`|Text file containing the locations of virtual stage and gauge points where simulated time histories of the water depth and discharge are recorded.|
|Parameter file|`.par`|Text file containing parameters to access various model features for running a simulation.|

### 2.1 Example

Consider an example of using the model to run a simulation of the Okushiri island test case in Section 3.3.2 of [Chowdhury et al., 2023](https://iwaponline.com/jh/article/doi/10.2166/hydro.2023.154/95732/GPU-parallelisation-of-Haar-wavelet-based-grid). This test case needs the following input files:

- `monai.par`
- `monai.dem`
- `monai.start`
- `monai.bci`
- `monai.bdy`
- `monai.stage`

To prepare these input files and run simulations of the test case, do the following steps:

1. Copy the executable to `gpu-mwdg2\cases`
2. Navigate to `gpu-mwdg2\cases\monai`
3. Prepare a stage file `monai.stage` by running `python stage.py` in a command prompt
4. Prepare the raster files `monai.dem` and `monai.start` by running `python raster.py`
5. Prepare the boundary inflow files `monai.bci` and `monai.bdy` by running `python inflow.py`
6. Prepare a parameter file called `monai.par` (parameters needed in file shown below)
7. Run `..\gpu-mwdg2.exe monai.par` in a command prompt to start running the simulation

#### Automated simulations

Instead of manually running the simulation by doing doing the steps above, automatically run the simulation by opening a command prompt at `gpu-mwdg2\cases\monai` and running `python simulate.py`. This will automatically do all the preprocessing for preparing the input files (steps 3 to 6), run several simulations (step 7), and postprocess the results. These `simulate.py` scripts are available for a number of test cases in the `gpu-mwdg2\cases` folder.

Doing steps 1 to 7 will give the following folder tree:

```
-- gpu-mwdg2
 |
 ...
 |
 | -- cases
    |
    ...
    |
    | -- monai
        |
        | -- results
           |
           ...
           |
        | monai.par
        | monai.dem
        | monai.start
        | monai.bci
        | monai.bdy
        | monai.stage
```

#### Parameter (`.par`) file for the example 

```
mwdg2
cumulative
refine_wall
ref_thickness 16
max_ref_lvl   9
epsilon       0.001
wall_height   0.5
initial_tstep 1
fpfric        0.01
sim_time      22.5
dirroot       results
massint       0.2
DEMfile       monai.dem
startfile     monai.start
stagefile     monai.stage
```

### 2.2 What is a parameter file?

A parameter (`.par`) file is a text file that specifies various parameters for running a simulation. The parameters and what function they serve are described in the table below.

| Parameter        | Description |
| ---------------- |-------------|
| `DEMfile       ` | Keyword followed by text. Specifies the name of the raster file containing the DEM. |
| `startfile     ` | Keyword followed by text. Specifies the name of the raster file containing the initial water depths. |
| `bcifile       ` | Keyword followed by text. Specifies the name of the boundary condition file. |
| `bdyfile       ` | Keyword followed by text. Specifies the name of the file containing the time-varying conditions. |
| `stagefile     ` | Keyword followed by text. Specifies the name of the stage file. |
| `sim_time      ` | Keyword followed by a decimal number. Specifies the simulation time in seconds. |
| `hwfv1         ` | Boolean keyword instructing the code to use the GPU-HWFV1 solver. |
| `mwdg2         ` | Boolean keyword instructing the code to use the GPU-MWDG2 solver. |
| `max_ref_lvl   ` | Keyword followed by an integer. Specifies the maximum refinement level L used by the model when setting the $2^L × 2^L$ finest resolution grid. For a test case involving a digital elevation model (DEM) made up of $N × M$ cells, the user must specify the value of L to be such that $2^L ≥ \max(N, M)$. By doing so, two areas emerge in the grid: the actual $N × M$ domain area defined by the DEM, and empty areas beyond the $N × M$ area where no DEM data are available and where no flow should should occur. |
| `epsilon       ` | Keyword followed by a decimal number. Specifies the error threshold ε used by the model to control the amount of coarsening in the non-uniform grid. A larger value allows a higher amount of coarsening. |
| `wall_height   ` | Keyword followed by a decimal number. Specifies the height of the wall in metres used by the model to fill the empty areas beyond the domain area. The user must specify a sufficiently high wall height to prevent any flow from exiting past the walls. |
| `refine_wall   ` | Boolean keyword to prevent the model from excessively coarsening the non-uniform grid around the wall separating the domain area from the void areas. The user can enable this refinement to ensure that very coarse cells around the wall do not lead to unrealistic calculations. |
| `ref_thickness ` | Keyword followed by an integer. Specifies the number of cells that are to be refined around the wall by the model when the refine_wall keyword is also specified. |
| `initial_t_step` | Keyword followed by a decimal number. Specifies the initial timestep. |
| `cumulative    ` | Boolean keyword instructing the model to produce a file called `cumulative-data.csv` that contains time series data designed to help in assessing the effect of grid adaptation on the runtime. |
| `vtk           ` | Boolean keyword to enable the output of `.vtk` output files for viewing flow and topography data over a non-uniform grid. |
| `c_prop        ` | Boolean keyword to enable the output of discharges; only applicable for quiescent test cases. |
| `raster_out    ` | Boolean keyword to enable the output of raster files. |
| `voutput_stage ` | Boolean keyword to also save the velocity time series at the stage locations. |
| `resroot       ` | Keyword followed by text. Specifies the prefix of the names of the output files. |
| `dirroot       ` | Keyword followed by text. Specifies the name of the folder where the output files are saved. |
| `massint       ` | Keyword followed by a decimal number. Specifies the time interval in seconds at which stage or guage data are recorded. |
| `saveint       ` | Keyword followed by a decimal number. Specifies the time interval in seconds at which raster or `.vtk` output files are saved. |
