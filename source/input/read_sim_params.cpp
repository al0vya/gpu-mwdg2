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
	    
	    	real cell_size = C(0.0);
	    
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
	    
	    	fp = fopen(dem_filename, "r");
	    
	    	if (NULL == fp)
	    	{
	    		fprintf(stderr, "Error opening DEM file for reading study area dimensions.\n");
	    		exit(-1);
	    	}
	    
	    	fscanf(fp, "%s %d", buf, &sim_params.xsz);
	    	fscanf(fp, "%s %d", buf, &sim_params.ysz);
	    	fscanf(fp, "%s %" NUM_FRMT, buf, &sim_params.xmin);
	    	fscanf(fp, "%s %" NUM_FRMT, buf, &sim_params.ymin);
	    	fscanf(fp, "%s %" NUM_FRMT, buf, &cell_size);
	    
	    	fclose(fp);
	    
	    	sim_params.xmax = sim_params.xmin + sim_params.xsz * cell_size;
	    	sim_params.ymax = sim_params.ymin + sim_params.ysz * cell_size;
	    
	    	fp = fopen(input_filename, "r");
	    
	    	if (NULL == fp)
	    	{
	    		fprintf(stderr, "Error opening input file for reading simulation parameters.\n");
	    		exit(-1);
	    	}
	    
	    	while ( strncmp(buf, "g", 1) )
	    	{
	    		if ( NULL == fgets( str, sizeof(str), fp) )
	    		{
	    			fprintf(stderr, "Error reading input file for gravity constant.\n");
	    			fclose(fp);
	    			exit(-1);
	    		}
	    
	    		sscanf(str, "%s %" NUM_FRMT, buf, &sim_params.g);
	    	}
	    
	    	rewind(fp);
	    
	    	while ( strncmp(buf, "sim_time", 8) )
	    	{
	    		if ( NULL == fgets( str, sizeof(str), fp) )
	    		{
	    			fprintf(stderr, "Error reading input file for simulation time.\n");
	    			fclose(fp);
	    			exit(-1);
	    		}
	    
	    		sscanf(str, "%s %" NUM_FRMT, buf, &sim_params.time);
	    	}
	    
	    	rewind(fp);
	    
	    	while ( strncmp(buf, "fpfric", 6) )
	    	{
	    		if ( NULL == fgets( str, sizeof(str), fp) )
	    		{
	    			fprintf(stderr, "Error reading input file for Manning coefficient.\n");
	    			fclose(fp);
	    			exit(-1);
	    		}
	    
	    		sscanf(str, "%s %" NUM_FRMT, buf, &sim_params.manning);
	    	}
	    
	    	fclose(fp);
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
	    	sim_params.time = C(100.0);
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
	    	sim_params.time = C(2.5);
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
	    	sim_params.time = C(1.3);
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
	    	sim_params.time = C(1.3);
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
	    	sim_params.time = C(10.0);
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
	    	sim_params.time = C(29.6);
	    	sim_params.manning = C(0.0125);
	    	break;
	    case 17: // parabolic bowl, period "T" = 14.4 s
	    case 18:
			real period = C(14.4);
			sim_params.xmin = C(-50.0);
	    	sim_params.xmax = C( 50.0);
	    	sim_params.ymin = C(-50.0);
	    	sim_params.ymax = C( 50.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
			sim_params.time = period * 2;
	    	sim_params.manning = C(0.0);
	    	break;
	    case 19: // three cones
	    	sim_params.xmin = C(10.0);
	    	sim_params.xmax = C(70.0);
	    	sim_params.ymin = C(-10.0);
	    	sim_params.ymax = C(50.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
	    	sim_params.time = C(100.0);
	    	sim_params.manning = C(0.0);
	    	break;
	    case 20: // diff and non diff topo c prop
	    case 21:
	    	sim_params.xmin = C(0.0);
	    	sim_params.xmax = C(75.0);
	    	sim_params.ymin = C(0.0);
	    	sim_params.ymax = C(75.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
	    	sim_params.time = C(100.0);
	    	sim_params.manning = C(0.0);
	    	break;
	    case 22: // radial dam break
	    	sim_params.xmin = C(-20.0);
	    	sim_params.xmax = C( 20.0);
	    	sim_params.ymin = C(-20.0);
	    	sim_params.ymax = C( 20.0);
	    	sim_params.xsz  = mesh_dim;
	    	sim_params.ysz  = mesh_dim;
	    	sim_params.g    = C(9.80665);
	    	sim_params.time = C(3.5);
	    	sim_params.manning = C(0.0);
	    	break;
	    default:
	    	break;
	}

	return sim_params;
}