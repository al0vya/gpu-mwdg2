#pragma once

#include "../types/Coordinate.h"
#include "../types/real.h"

#include "../classes/SolverParams.h"

#include "../input/read_keyword_str.h"

typedef struct SimulationParams
{
	real       xmin      = C(0.0);
	real       xmax      = C(0.0);
	real       ymin      = C(0.0);
	real       ymax      = C(0.0);
	Coordinate xsz       = 0;
	Coordinate ysz       = 0;
	real       g         = C(9.80665);
	real       time      = C(0.0);
	real       manning   = C(0.0);
	bool       is_monai  = false;
	bool       is_oregon = false;
    
    SimulationParams
    (
        const int&  test_case,
        const char* input_filename,
        const int&  max_ref_lvl
    )
    {
        int mesh_dim = 1 << max_ref_lvl;
        
        switch (test_case)
        {
            case 0: // raster file based test case
            {
                char dem_filename_buf[128] = {'\0'};
                read_keyword_str(input_filename, "DEMfile", 7, dem_filename_buf);
    
                FILE* fp = fopen(dem_filename_buf, "r");
    
                if (NULL == fp)
                {
                    fprintf(stderr, "Error opening DEM file for reading simulation parameters, file: %s, line: %d.\n", __FILE__, __LINE__);
                    exit(-1);
                }
    
                real cellsize = C(0.0);
    
                this->xsz       = read_keyword_int (dem_filename_buf, "ncols", 5);
                this->ysz       = read_keyword_int (dem_filename_buf, "nrows", 5);
                this->xmin      = read_keyword_real(dem_filename_buf, "xllcorner", 9);
                this->ymin      = read_keyword_real(dem_filename_buf, "yllcorner", 9);
                cellsize        = read_keyword_real(dem_filename_buf, "cellsize", 8);
                this->xmax      = this->xmin + this->xsz * cellsize;
                this->ymax      = this->ymin + this->ysz * cellsize;
                this->time      = read_keyword_real(input_filename, "sim_time", 8);
                this->manning   = read_keyword_real(input_filename, "fpfric", 6);
                this->is_monai  = !strncmp("monai.dem", dem_filename_buf, 9);
                this->is_oregon = !strncmp("oregon-seaside-0p02m.dem", dem_filename_buf, 24);
            }
                break;
            case 1: // c prop
            case 2:
            case 3:
            case 4:
                this->xmin = C(0.0);
                this->xmax = C(50.0);
                this->ymin = C(0.0);
                this->ymax = C(50.0);
                this->xsz  = mesh_dim;
                this->ysz  = mesh_dim;
                this->g    = C(9.80665);
                this->time = read_keyword_real(input_filename, "sim_time", 8);
                this->manning = C(0.0);
                break;
            case 5: // wet dam break
            case 6: 
                this->xmin = C(0.0);
                this->xmax = C(50.0);
                this->ymin = C(0.0);
                this->ymax = C(50.0);
                this->xsz  = mesh_dim;
                this->ysz  = mesh_dim;
                this->g    = C(9.80665);
                this->time = read_keyword_real(input_filename, "sim_time", 8);
                this->manning = C(0.0);
                break;
            case 7: // dry dam break
            case 8:
                this->xmin = C(0.0);
                this->xmax = C(50.0);
                this->ymin = C(0.0);
                this->ymax = C(50.0);
                this->xsz  = mesh_dim;
                this->ysz  = mesh_dim;
                this->g    = C(9.80665);
                this->time = read_keyword_real(input_filename, "sim_time", 8);
                this->manning = C(0.0);
                break;
            case 9: // dry dam break w fric
            case 10:
                this->xmin = C(0.0);
                this->xmax = C(50.0);
                this->ymin = C(0.0);
                this->ymax = C(50.0);
                this->xsz  = mesh_dim;
                this->ysz  = mesh_dim;
                this->g    = C(9.80665);
                this->time = read_keyword_real(input_filename, "sim_time", 8);
                this->manning = C(0.02);
                break;
            case 11: // building overtopping
            case 12:
            case 13:
            case 14:
                this->xmin = C(0.0);
                this->xmax = C(50.0);
                this->ymin = C(0.0);
                this->ymax = C(50.0);
                this->xsz  = mesh_dim;
                this->ysz  = mesh_dim;
                this->g    = C(9.80665);
                this->time = read_keyword_real(input_filename, "sim_time", 8);
                this->manning = C(0.02);
                break;
            case 15: // triangle dam break
            case 16:
                this->xmin = C(0.0);
                this->xmax = C(38.0);
                this->ymin = C(0.0);
                this->ymax = C(38.0);
                this->xsz  = mesh_dim;
                this->ysz  = mesh_dim;
                this->g    = C(9.80665);
                this->time = read_keyword_real(input_filename, "sim_time", 8);
                this->manning = C(0.0125);
                break;
            case 17: // parabolic bowl, period "T" = 14.4 s
            case 18:
                this->xmin = C(-50.0);
                this->xmax = C( 50.0);
                this->ymin = C(-50.0);
                this->ymax = C( 50.0);
                this->xsz  = mesh_dim;
                this->ysz  = mesh_dim;
                this->g    = C(9.80665);
                this->time = read_keyword_real(input_filename, "sim_time", 8);
                this->manning = C(0.0);
                break;
            case 19: // three cones c prop
                this->xmin = C( 10.0);
                this->xmax = C( 70.0);
                this->ymin = C(-10.0);
                this->ymax = C( 50.0);
                this->xsz  = mesh_dim;
                this->ysz  = mesh_dim;
                this->g    = C(9.80665);
                this->time = read_keyword_real(input_filename, "sim_time", 8);
                this->manning = C(0.0);
                break;
            case 20: // three cones dam break
                this->xmin = C(0.0);
                this->xmax = C(70.0);
                this->ymin = C(0.0);
                this->ymax = C(30.0);
                this->xsz  = mesh_dim;
                this->ysz  = mesh_dim;
                this->g    = C(9.80665);
                this->time = read_keyword_real(input_filename, "sim_time", 8);
                this->manning = C(0.018);
                break;
            case 21: // diff and non diff topo c prop
            case 22:
                this->xmin = C(0.0);
                this->xmax = C(75.0);
                this->ymin = C(0.0);
                this->ymax = C(75.0);
                this->xsz  = mesh_dim;
                this->ysz  = mesh_dim;
                this->g    = C(9.80665);
                this->time = read_keyword_real(input_filename, "sim_time", 8);
                this->manning = C(0.0);
                break;
            case 23: // radial dam break
                this->xmin = C(-20.0);
                this->xmax = C( 20.0);
                this->ymin = C(-20.0);
                this->ymax = C( 20.0);
                this->xsz  = mesh_dim;
                this->ysz  = mesh_dim;
                this->g    = C(9.80665);
                this->time = read_keyword_real(input_filename, "sim_time", 8);
                this->manning = C(0.0);
                break;
            default:
                break;
        }
    }

} SimulationParams;