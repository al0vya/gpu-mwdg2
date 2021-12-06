#include "read_solver_params.h"

SolverParameters read_solver_params
(
	const char* input_filename
)
{
	FILE* fp = fopen(input_filename, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening input file when reading solver parameters.\n");
		exit(-1);
	}

	SolverParameters solver_params = SolverParameters();

	char str[255]            = {'\0'};
	char buf[64]             = {'\0'};
	char solvertype_buf[255] = {'\0'};

	while ( strncmp(buf, "max_ref_lvl", 11) )
	{
		if ( NULL == fgets( str, sizeof(str), fp) )
		{
			fprintf(stderr, "Error reading input file for maximum refinement level.\n");
			fclose(fp);
			exit(-1);
		}

		sscanf(str, "%s %d", buf, &solver_params.L);
	}

	rewind(fp);
	
	while ( strncmp(buf, "min_dt", 6) )
	{
		if ( NULL == fgets( str, sizeof(str), fp) )
		{
			fprintf(stderr, "Error reading input file for maximum refinement level.\n");
			fclose(fp);
			exit(-1);
		}

		sscanf(str, "%s %" NUM_FRMT, buf, &solver_params.min_dt);
	}

	rewind(fp);

	while ( strncmp(buf, "epsilon", 7) )
	{
		if ( NULL == fgets( str, sizeof(str), fp) )
		{
			fprintf(stderr, "Error reading input file for epsilon.\n");
			fclose(fp);
			exit(-1);
		}

		sscanf(str, "%s %" NUM_FRMT, buf, &solver_params.epsilon);
	}

	rewind(fp);

	while ( strncmp(buf, "tol_h", 5) )
	{
		if ( NULL == fgets( str, sizeof(str), fp) )
		{
			fprintf(stderr, "Error reading input file for tol h.\n");
			fclose(fp);
			exit(-1);
		}

		sscanf(str, "%s %" NUM_FRMT, buf, &solver_params.tol_h);
	}

	rewind(fp);

	while ( strncmp(buf, "tol_q", 5) )
	{
		if ( NULL == fgets( str, sizeof(str), fp) )
		{
			fprintf(stderr, "Error reading input file for tol q.\n");
			fclose(fp);
			exit(-1);
		}

		sscanf(str, "%s %" NUM_FRMT, buf, &solver_params.tol_q);
	}

	rewind(fp);

	while ( strncmp(buf, "tol_s", 5) )
	{
		if ( NULL == fgets( str, sizeof(str), fp) )
		{
			fprintf(stderr, "Error reading input file for tol speed.\n");
			fclose(fp);
			exit(-1);
		}

		sscanf(str, "%s %" NUM_FRMT, buf, &solver_params.tol_s);
	}

	rewind(fp);

	while ( strncmp(buf, "solver", 6) )
	{
		if ( NULL == fgets( str, sizeof(str), fp) )
		{
			fprintf(stderr, "Error reading input file for solver type.\n");
			fclose(fp);
			exit(-1);
		}

		sscanf(str, "%s %s", buf, solvertype_buf);
	}

	if ( !strncmp(solvertype_buf, "hw", 2) )
	{
		solver_params.solver_type = HWFV1;
		solver_params.CFL         = C(0.5);
	}
	else if ( !strncmp(solvertype_buf, "mw", 2) )
	{
		solver_params.solver_type = MWDG2;
		solver_params.CFL         = C(0.33);
	}
	else
	{
		fprintf(stderr, "Error: invalid solver type specified, please specify either \"hw\" or \"mw\".\n");
		fclose(fp);
		exit(-1);
	}

	rewind(fp);

	while ( strncmp(buf, "wall_height", 11) )
	{
		if ( NULL == fgets(str, sizeof(str), fp) )
		{
			fprintf(stderr, "Error reading input file for boundary wall height.\n");
			fclose(fp);
			exit(-1);
		}

		sscanf(str, "%s %" NUM_FRMT, buf, &solver_params.wall_height);
	}

	fclose(fp);

	return solver_params;
}