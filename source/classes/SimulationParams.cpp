#include "SimulationParams.h"

SimulationParams::SimulationParams() = default;

SimulationParams::SimulationParams
(
    const int&  test_case,
    const char* input_filename,
    const int&  max_ref_lvl
)
    : test_case(test_case)
{
    int mesh_dim = 1 << max_ref_lvl;
    
    switch (this->test_case)
    {
        case 0: // raster file based test case
        {
            char dem_filename_buf[128] = {'\0'};
            read_keyword_str(input_filename, "DEMfile", dem_filename_buf);

            this->xsz       = read_keyword_int (dem_filename_buf, "ncols");
            this->ysz       = read_keyword_int (dem_filename_buf, "nrows");
            this->xmin      = read_keyword_real(dem_filename_buf, "xllcorner");
            this->ymin      = read_keyword_real(dem_filename_buf, "yllcorner");
            real cellsize   = read_keyword_real(dem_filename_buf, "cellsize");
            this->xmax      = this->xmin + this->xsz * cellsize;
            this->ymax      = this->ymin + this->ysz * cellsize;
            this->time      = read_keyword_real(input_filename, "sim_time");
            this->manning   = read_keyword_real(input_filename, "fpfric");
            this->is_monai  = read_keyword_bool(input_filename, "monai");
            this->is_oregon = read_keyword_bool(input_filename, "oregon-seaside");
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
            this->time = read_keyword_real(input_filename, "sim_time");
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
            this->time = read_keyword_real(input_filename, "sim_time");
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
            this->time = read_keyword_real(input_filename, "sim_time");
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
            this->time = read_keyword_real(input_filename, "sim_time");
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
            this->time = read_keyword_real(input_filename, "sim_time");
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
            this->time = read_keyword_real(input_filename, "sim_time");
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
            this->time = read_keyword_real(input_filename, "sim_time");
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
            this->time = read_keyword_real(input_filename, "sim_time");
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
            this->time = read_keyword_real(input_filename, "sim_time");
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
            this->time = read_keyword_real(input_filename, "sim_time");
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
            this->time = read_keyword_real(input_filename, "sim_time");
            this->manning = C(0.0);
            break;
        default:
            break;
    }
}