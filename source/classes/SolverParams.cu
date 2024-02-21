#include "SolverParams.h"

SolverParams::SolverParams() = default;

SolverParams::SolverParams
(
    const char* input_filename
)
{
    this->L             = read_keyword_int (input_filename, "max_ref_lvl");
    this->initial_tstep = read_keyword_real(input_filename, "initial_tstep");
    this->epsilon       = read_keyword_real(input_filename, "epsilon");
    this->wall_height   = read_keyword_real(input_filename, "wall_height");

    if ( read_keyword_bool(input_filename, "hwfv1") )
    {
        this->solver_type = HWFV1;
        this->CFL         = C(0.5);
    }
    else if ( read_keyword_bool(input_filename, "mwdg2") )
    {
        this->solver_type = MWDG2;
        this->CFL         = C(0.3);
    }
    else
    {
        fprintf(stderr, "Error: invalid adaptive solver type specified, please specify either \"hwfv1\" or \"mwdg2\", file: %s, line: %d.\n", __FILE__, __LINE__);
        exit(-1);
    }

    this->grading = read_keyword_bool(input_filename, "grading");
    
    this->limitslopes = read_keyword_bool(input_filename, "limitslopes");
    
    if (this->limitslopes)
    {
        this->tol_Krivo = read_keyword_real(input_filename, "tol_Krivo");
    }
    
    this->refine_wall = read_keyword_bool(input_filename, "refine_wall");
    
    if (this->refine_wall)
    {
        this->ref_thickness = read_keyword_int(input_filename, "ref_thickness");
    }

    this->startq2d = read_keyword_bool(input_filename, "startq2d");
}