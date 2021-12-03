#include "write_mesh_info.h"

void write_mesh_info
(
	const SimulationParameters& sim_params,
	const int&                  mesh_dim,
	const char*                 respath
)
{
	char fullpath[255];

	sprintf(fullpath, "%s%s", respath, "mesh_info.csv");
	
	FILE* fp = fopen(fullpath, "w");

	if (NULL == fp)
	{
		fprintf(stderr, "Error in opening mesh information file.");
		exit(-1);
	}

	fprintf(fp, "mesh_dim,xmin,xmax,ymin,ymax,xsz,ysz\n");

	fprintf
	(
		fp,
		"%d,%" NUM_FRMT ",%" NUM_FRMT ",%" NUM_FRMT ",%" NUM_FRMT ",%d,%d\n", 
		mesh_dim, 
		sim_params.xmin, 
		sim_params.xmax, 
		sim_params.ymin, 
		sim_params.ymax,
		sim_params.xsz,
		sim_params.ysz
	);

	fclose(fp);
}