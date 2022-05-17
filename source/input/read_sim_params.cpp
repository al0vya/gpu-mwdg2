#include "read_sim_params.h"

SimulationParams read_sim_params
(
	const int&              test_case,
	const char*             input_filename,
	const SolverParams& solver_params
)
{
	int mesh_dim = 1 << solver_params.L;
	
	SimulationParams sim_params = SimulationParams();

	switch (test_case)
	{
	    case 0: // raster file based test case
	    {
	    	char str[255]         = {'\0'};
	    	char buf[64]          = {'\0'};
	    	char rasterroot[64]   = {'\0'};
	    	char dem_filename[64] = {'\0'};
	    
	    	real cellsize = C(0.0);
	    
	    	FILE* fp = fopen(input_filename, "r");
	    
	    	if (NULL == fp)
	    	{
	    		fprintf(stderr, "Error opening input file for reading study area dimensions, file: %s, line: %d.\n", __FILE__, __LINE__);
	    		exit(-1);
	    	}
	    
	    	while ( strncmp(buf, "rasterroot", 10) )
	    	{
	    		if ( NULL == fgets( str, sizeof(str), fp) )
	    		{
	    			fprintf(stderr, "Error reading input file for reading raster root filename, file: %s, line: %d.\n", __FILE__, __LINE__);
	    			fclose(fp);
	    			exit(-1);
	    		}
	    
	    		sscanf(str, "%s %s", buf, rasterroot);
	    	}
	    
	    	fclose(fp);
	    
	    	strcpy(dem_filename, rasterroot);
	    	strcat(dem_filename, ".dem");

			sim_params.xsz      = read_keyword_int (dem_filename, "ncols", 5);
			sim_params.ysz      = read_keyword_int (dem_filename, "nrows", 5);
			sim_params.xmin     = read_keyword_real(dem_filename, "xllcorner", 9);
			sim_params.ymin     = read_keyword_real(dem_filename, "yllcorner", 9);
			cellsize            = read_keyword_real(dem_filename, "cellsize", 8);
	    	sim_params.xmax     = sim_params.xmin + sim_params.xsz * cellsize;
	    	sim_params.ymax     = sim_params.ymin + sim_params.ysz * cellsize;
			sim_params.g        = read_keyword_real(input_filename, "g", 1);
			sim_params.time     = read_keyword_real(input_filename, "sim_time", 8);
			sim_params.manning  = read_keyword_real(input_filename, "fpfric", 6);
			sim_params.is_monai = !strncmp("monai", rasterroot, 5);
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