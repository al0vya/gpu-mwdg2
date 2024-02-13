#pragma once

#include "../types/real.h"
#include "../types/SolverTypes.h"

#include "../input/read_keyword_int.h"
#include "../input/read_keyword_bool.h"
#include "../input/read_keyword_real.h"

typedef struct SolverParams
{
	int  L             = 0;
	real initial_tstep = C(0.0);
	real epsilon       = C(0.0);
	real tol_h         = C(1e-3);
	real tol_q         = C(0.0);
	real tol_s         = C(1e-9);
	real wall_height   = C(0.0);
	int  solver_type   = 0;
	real CFL           = C(0.0);
	bool grading       = false;
	bool limitslopes   = false;
	real tol_Krivo     = C(9999.0);
	bool refine_wall   = false;
	int  ref_thickness = 0;
	bool startq2d      = false;
    
    SolverParams
    (
        const char* input_filename
    )
    {
        this->L             = read_keyword_int (input_filename, "max_ref_lvl", 11);
        this->initial_tstep = read_keyword_real(input_filename, "initial_tstep", 13);
        this->epsilon       = read_keyword_real(input_filename, "epsilon", 7);
        this->wall_height   = read_keyword_real(input_filename, "wall_height", 11);
    
        if ( read_keyword_bool(input_filename, "hwfv1", 5) )
        {
            this->solver_type = HWFV1;
            this->CFL         = C(0.5);
        }
        else if ( read_keyword_bool(input_filename, "mwdg2", 5) )
        {
            this->solver_type = MWDG2;
            this->CFL         = C(0.3);
        }
        else
        {
            fprintf(stderr, "Error: invalid adaptive solver type specified, please specify either \"hwfv1\" or \"mwdg2\", file: %s, line: %d.\n", __FILE__, __LINE__);
            exit(-1);
        }
    
        this->grading = read_keyword_bool(input_filename, "grading", 7);
        
        this->limitslopes = read_keyword_bool(input_filename, "limitslopes", 11);
        
        if (this->limitslopes)
        {
            this->tol_Krivo = read_keyword_real(input_filename, "tol_Krivo", 9);
        }
        
        this->refine_wall = read_keyword_bool(input_filename, "refine_wall", 11);
        
        if (this->refine_wall)
        {
            this->ref_thickness = read_keyword_int(input_filename, "ref_thickness", 13);
        }
    
        this->startq2d = read_keyword_bool(input_filename, "startq2d", 8);
    }

} SolverParams;