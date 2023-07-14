#include "read_sim_params.h"

SimulationParams read_sim_params
(
	const int&          test_case,
	const char*         input_filename,
	const SolverParams& solver_params
)
{
	int mesh_dim = 1 << solver_params.L;
	
	SimulationParams sim_params = SimulationParams();

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

			sim_params.xsz       = read_keyword_int (dem_filename_buf, "ncols", 5);
			sim_params.ysz       = read_keyword_int (dem_filename_buf, "nrows", 5);
			sim_params.xmin      = read_keyword_real(dem_filename_buf, "xllcorner", 9);
			sim_params.ymin      = read_keyword_real(dem_filename_buf, "yllcorner", 9);
			cellsize             = read_keyword_real(dem_filename_buf, "cellsize", 8);
	    	sim_params.xmax      = sim_params.xmin + sim_params.xsz * cellsize;
	    	sim_params.ymax      = sim_params.ymin + sim_params.ysz * cellsize;
			sim_params.time      = read_keyword_real(input_filename, "sim_time", 8);
			sim_params.manning   = read_keyword_real(input_filename, "fpfric", 6);
			sim_params.is_monai  = !strncmp("monai.dem", dem_filename_buf, 9);
			sim_params.is_oregon = !strncmp("oregon-seaside.dem", dem_filename_buf, 18);
	    }
	    	break;
	    case 1: // c prop
	    case 2:
	    case 3:
	    case 4:
	    	sim_params.xmin = C(0.0);
	    	sim_params.xmax = C(50.0);
	    	sim_params.ymin = C(0.0);
	    	sim_params.ymax = C(50.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
	    	sim_params.time = read_keyword_real(input_filename, "sim_time", 8);
	    	sim_params.manning = C(0.0);
	    	break;
	    case 5: // wet dam break
	    case 6: 
	    	sim_params.xmin = C(0.0);
	    	sim_params.xmax = C(50.0);
	    	sim_params.ymin = C(0.0);
	    	sim_params.ymax = C(50.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
			sim_params.time = read_keyword_real(input_filename, "sim_time", 8);
			sim_params.manning = C(0.0);
	    	break;
	    case 7: // dry dam break
	    case 8:
	    	sim_params.xmin = C(0.0);
	    	sim_params.xmax = C(50.0);
	    	sim_params.ymin = C(0.0);
	    	sim_params.ymax = C(50.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
			sim_params.time = read_keyword_real(input_filename, "sim_time", 8);
			sim_params.manning = C(0.0);
	    	break;
	    case 9: // dry dam break w fric
	    case 10:
	    	sim_params.xmin = C(0.0);
	    	sim_params.xmax = C(50.0);
	    	sim_params.ymin = C(0.0);
	    	sim_params.ymax = C(50.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
			sim_params.time = read_keyword_real(input_filename, "sim_time", 8);
			sim_params.manning = C(0.02);
	    	break;
	    case 11: // building overtopping
	    case 12:
	    case 13:
	    case 14:
	    	sim_params.xmin = C(0.0);
	    	sim_params.xmax = C(50.0);
	    	sim_params.ymin = C(0.0);
	    	sim_params.ymax = C(50.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
			sim_params.time = read_keyword_real(input_filename, "sim_time", 8);
			sim_params.manning = C(0.02);
	    	break;
	    case 15: // triangle dam break
	    case 16:
	    	sim_params.xmin = C(0.0);
	    	sim_params.xmax = C(38.0);
	    	sim_params.ymin = C(0.0);
	    	sim_params.ymax = C(38.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
	    	sim_params.time = read_keyword_real(input_filename, "sim_time", 8);
	    	sim_params.manning = C(0.0125);
	    	break;
	    case 17: // parabolic bowl, period "T" = 14.4 s
	    case 18:
			sim_params.xmin = C(-50.0);
	    	sim_params.xmax = C( 50.0);
	    	sim_params.ymin = C(-50.0);
	    	sim_params.ymax = C( 50.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
			sim_params.time = read_keyword_real(input_filename, "sim_time", 8);
			sim_params.manning = C(0.0);
	    	break;
	    case 19: // three cones c prop
	    	sim_params.xmin = C( 10.0);
	    	sim_params.xmax = C( 70.0);
	    	sim_params.ymin = C(-10.0);
	    	sim_params.ymax = C( 50.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
			sim_params.time = read_keyword_real(input_filename, "sim_time", 8);
			sim_params.manning = C(0.0);
	    	break;
	    case 20: // three cones dam break
	    	sim_params.xmin = C(0.0);
	    	sim_params.xmax = C(70.0);
	    	sim_params.ymin = C(0.0);
	    	sim_params.ymax = C(30.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
			sim_params.time = read_keyword_real(input_filename, "sim_time", 8);
			sim_params.manning = C(0.018);
	    	break;
	    case 21: // diff and non diff topo c prop
	    case 22:
	    	sim_params.xmin = C(0.0);
	    	sim_params.xmax = C(75.0);
	    	sim_params.ymin = C(0.0);
	    	sim_params.ymax = C(75.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
			sim_params.time = read_keyword_real(input_filename, "sim_time", 8);
			sim_params.manning = C(0.0);
	    	break;
	    case 23: // radial dam break
	    	sim_params.xmin = C(-20.0);
	    	sim_params.xmax = C( 20.0);
	    	sim_params.ymin = C(-20.0);
	    	sim_params.ymax = C( 20.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
			sim_params.time = read_keyword_real(input_filename, "sim_time", 8);
			sim_params.manning = C(0.0);
	    	break;
	    default:
	    	break;
	}

	return sim_params;
}