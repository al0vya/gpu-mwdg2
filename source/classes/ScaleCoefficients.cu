#include "ScaleCoefficients.h"

ScaleCoefficients::ScaleCoefficients
(
    const SolverParams& solver_params
)
    : levels(solver_params.L), solver_type(solver_params.solver_type)
{
    const int num_scale_coeffs = get_lvl_idx(this->levels + 1);
	
	const size_t bytes = sizeof(real) * num_scale_coeffs;
	
	this->eta0 = (real*)malloc_device(bytes);
	this->qx0  = (real*)malloc_device(bytes);
	this->qy0  = (real*)malloc_device(bytes);
	this->z0   = (real*)malloc_device(bytes);

	this->eta1x = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
	this->qx1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
	this->qy1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
	this->z1x   = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;

	this->eta1y = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
	this->qx1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
	this->qy1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
	this->z1y   = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
}

ScaleCoefficients::ScaleCoefficients
(
	const SolverParams& solver_params,
	const char*         dirroot,
	const char*         prefix
)
    : levels(solver_params.L), solver_type(solver_params.solver_type)
{
	if (this->solver_type == HWFV1)
	{
		char filename_eta0[255] = {'\0'};
		char filename_qx0[255]  = {'\0'};
		char filename_qy0[255]  = {'\0'};
		char filename_z0[255]   = {'\0'};

		sprintf(filename_eta0, "%s%s", prefix, "-scale-coeffs-eta0-hw");
		sprintf(filename_qx0,  "%s%s", prefix, "-scale-coeffs-qx0-hw");
		sprintf(filename_qy0,  "%s%s", prefix, "-scale-coeffs-qy0-hw");
		sprintf(filename_z0,   "%s%s", prefix, "-scale-coeffs-z0-hw");

		this->eta0 = read_hierarchy_array_real(this->levels, dirroot, filename_eta0);
		this->qx0  = read_hierarchy_array_real(this->levels, dirroot, filename_qx0);
		this->qy0  = read_hierarchy_array_real(this->levels, dirroot, filename_qy0);
		this->z0   = read_hierarchy_array_real(this->levels, dirroot, filename_z0);
	}
	else if (this->solver_type == MWDG2)
	{
		char filename_eta0[255]  = {'\0'};
		char filename_qx0[255]   = {'\0'};
		char filename_qy0[255]   = {'\0'};
		char filename_z0[255]    = {'\0'};
		char filename_eta1x[255] = {'\0'};
		char filename_qx1x[255]  = {'\0'};
		char filename_qy1x[255]  = {'\0'};
		char filename_z1x[255]   = {'\0'};
		char filename_eta1y[255] = {'\0'};
		char filename_qx1y[255]  = {'\0'};
		char filename_qy1y[255]  = {'\0'};
		char filename_z1y[255]   = {'\0'};

		sprintf(filename_eta0,  "%s%s", prefix, "-scale-coeffs-eta0-mw");
		sprintf(filename_qx0,   "%s%s", prefix, "-scale-coeffs-qx0-mw");
		sprintf(filename_qy0,   "%s%s", prefix, "-scale-coeffs-qy0-mw");
		sprintf(filename_z0,    "%s%s", prefix, "-scale-coeffs-z0-mw");
		sprintf(filename_eta1x, "%s%s", prefix, "-scale-coeffs-eta1x-mw");
		sprintf(filename_qx1x,  "%s%s", prefix, "-scale-coeffs-qx1x-mw");
		sprintf(filename_qy1x,  "%s%s", prefix, "-scale-coeffs-qy1x-mw");
		sprintf(filename_z1x,   "%s%s", prefix, "-scale-coeffs-z1x-mw");
		sprintf(filename_eta1y, "%s%s", prefix, "-scale-coeffs-eta1y-mw");
		sprintf(filename_qx1y,  "%s%s", prefix, "-scale-coeffs-qx1y-mw");
		sprintf(filename_qy1y,  "%s%s", prefix, "-scale-coeffs-qy1y-mw");
		sprintf(filename_z1y,   "%s%s", prefix, "-scale-coeffs-z1y-mw");
		
		this->eta0  = read_hierarchy_array_real(this->levels, dirroot, filename_eta0);
		this->qx0   = read_hierarchy_array_real(this->levels, dirroot, filename_qx0); 
		this->qy0   = read_hierarchy_array_real(this->levels, dirroot, filename_qy0); 
		this->z0    = read_hierarchy_array_real(this->levels, dirroot, filename_z0);  
		this->eta1x = read_hierarchy_array_real(this->levels, dirroot, filename_eta1x);
		this->qx1x  = read_hierarchy_array_real(this->levels, dirroot, filename_qx1x);
		this->qy1x  = read_hierarchy_array_real(this->levels, dirroot, filename_qy1x);
		this->z1x   = read_hierarchy_array_real(this->levels, dirroot, filename_z1x); 
		this->eta1y = read_hierarchy_array_real(this->levels, dirroot, filename_eta1y);
		this->qx1y  = read_hierarchy_array_real(this->levels, dirroot, filename_qx1y);
		this->qy1y  = read_hierarchy_array_real(this->levels, dirroot, filename_qy1y);
		this->z1y   = read_hierarchy_array_real(this->levels, dirroot, filename_z1y); 
	}
}

ScaleCoefficients::ScaleCoefficients(const ScaleCoefficients& original)
{
	*this        = original;
	is_copy_cuda = true;
}

ScaleCoefficients::~ScaleCoefficients()
{
    if (!is_copy_cuda)
	{
		CHECK_CUDA_ERROR( free_device(eta0) );
		CHECK_CUDA_ERROR( free_device(qx0) );
		CHECK_CUDA_ERROR( free_device(qy0) );
		CHECK_CUDA_ERROR( free_device(z0) );

		CHECK_CUDA_ERROR( free_device(eta1x) );
		CHECK_CUDA_ERROR( free_device(qx1x) );
		CHECK_CUDA_ERROR( free_device(qy1x) );
		CHECK_CUDA_ERROR( free_device(z1x) );

		CHECK_CUDA_ERROR( free_device(eta1y) );
		CHECK_CUDA_ERROR( free_device(qx1y) );
		CHECK_CUDA_ERROR( free_device(qy1y) );
		CHECK_CUDA_ERROR( free_device(z1y) );
	}
}

void ScaleCoefficients::write_to_file
(
	const char* dirroot,
	const char* prefix
)
{
    if (this->solver_type == HWFV1)
	{
		char filename_eta0[255] = {'\0'};
		char filename_qx0[255]  = {'\0'};
		char filename_qy0[255]  = {'\0'};
		char filename_z0[255]   = {'\0'};

		sprintf(filename_eta0, "%s%s", prefix, "-scale-coeffs-eta0-hw");
		sprintf(filename_qx0,  "%s%s", prefix, "-scale-coeffs-qx0-hw");
		sprintf(filename_qy0,  "%s%s", prefix, "-scale-coeffs-qy0-hw");
		sprintf(filename_z0,   "%s%s", prefix, "-scale-coeffs-z0-hw");
		
		write_hierarchy_array_real(dirroot, filename_eta0, this->eta0, this->levels);
		write_hierarchy_array_real(dirroot, filename_qx0,  this->qx0,  this->levels);
		write_hierarchy_array_real(dirroot, filename_qy0,  this->qy0,  this->levels);
		write_hierarchy_array_real(dirroot, filename_z0,   this->z0,   this->levels);
	}
	if (this->solver_type == MWDG2)
	{
		char filename_eta0[255]  = {'\0'};
		char filename_qx0[255]   = {'\0'};
		char filename_qy0[255]   = {'\0'};
		char filename_z0[255]    = {'\0'};
		char filename_eta1x[255] = {'\0'};
		char filename_qx1x[255]  = {'\0'};
		char filename_qy1x[255]  = {'\0'};
		char filename_z1x[255]   = {'\0'};
		char filename_eta1y[255] = {'\0'};
		char filename_qx1y[255]  = {'\0'};
		char filename_qy1y[255]  = {'\0'};
		char filename_z1y[255]   = {'\0'};

		sprintf(filename_eta0,  "%s%s", prefix, "-scale-coeffs-eta0-mw");
		sprintf(filename_qx0,   "%s%s", prefix, "-scale-coeffs-qx0-mw");
		sprintf(filename_qy0,   "%s%s", prefix, "-scale-coeffs-qy0-mw");
		sprintf(filename_z0,    "%s%s", prefix, "-scale-coeffs-z0-mw");
		sprintf(filename_eta1x, "%s%s", prefix, "-scale-coeffs-eta1x-mw");
		sprintf(filename_qx1x,  "%s%s", prefix, "-scale-coeffs-qx1x-mw");
		sprintf(filename_qy1x,  "%s%s", prefix, "-scale-coeffs-qy1x-mw");
		sprintf(filename_z1x,   "%s%s", prefix, "-scale-coeffs-z1x-mw");
		sprintf(filename_eta1y, "%s%s", prefix, "-scale-coeffs-eta1y-mw");
		sprintf(filename_qx1y,  "%s%s", prefix, "-scale-coeffs-qx1y-mw");
		sprintf(filename_qy1y,  "%s%s", prefix, "-scale-coeffs-qy1y-mw");
		sprintf(filename_z1y,   "%s%s", prefix, "-scale-coeffs-z1y-mw");
		
		write_hierarchy_array_real(dirroot, filename_eta0,  this->eta0,  this->levels);
		write_hierarchy_array_real(dirroot, filename_qx0,   this->qx0,   this->levels);
		write_hierarchy_array_real(dirroot, filename_qy0,   this->qy0,   this->levels);
		write_hierarchy_array_real(dirroot, filename_z0,    this->z0,    this->levels);
		write_hierarchy_array_real(dirroot, filename_eta1x, this->eta1x, this->levels);
		write_hierarchy_array_real(dirroot, filename_qx1x,  this->qx1x,  this->levels);
		write_hierarchy_array_real(dirroot, filename_qy1x,  this->qy1x,  this->levels);
		write_hierarchy_array_real(dirroot, filename_z1x,   this->z1x,   this->levels);
		write_hierarchy_array_real(dirroot, filename_eta1y, this->eta1y, this->levels);
		write_hierarchy_array_real(dirroot, filename_qx1y,  this->qx1y,  this->levels);
		write_hierarchy_array_real(dirroot, filename_qy1y,  this->qy1y,  this->levels);
		write_hierarchy_array_real(dirroot, filename_z1y,   this->z1y,   this->levels);
	}
}

real ScaleCoefficients::verify
(
	const char* dirroot,
	const char* prefix
)
{
	if (this->solver_type == HWFV1)
	{
		char filename_eta0[255] = {'\0'};
		char filename_qx0[255]  = {'\0'};
		char filename_qy0[255]  = {'\0'};
		char filename_z0[255]   = {'\0'};

		sprintf(filename_eta0, "%s%s", prefix, "-scale-coeffs-eta0-hw");
		sprintf(filename_qx0,  "%s%s", prefix, "-scale-coeffs-qx0-hw");
		sprintf(filename_qy0,  "%s%s", prefix, "-scale-coeffs-qy0-hw");
		sprintf(filename_z0,   "%s%s", prefix, "-scale-coeffs-z0-hw");
		
		real* d_eta0_verified = read_hierarchy_array_real(this->levels, dirroot, filename_eta0);
		real* d_qx0_verified  = read_hierarchy_array_real(this->levels, dirroot, filename_qx0);
		real* d_qy0_verified  = read_hierarchy_array_real(this->levels, dirroot, filename_qy0);
		real* d_z0_verified   = read_hierarchy_array_real(this->levels, dirroot, filename_z0);

		const int num_scale_coeffs = get_lvl_idx(this->levels + 1);

		real error_eta0 = compute_error(this->eta0, d_eta0_verified, num_scale_coeffs);
		real error_qx0  = compute_error(this->qx0,  d_qx0_verified,  num_scale_coeffs);
		real error_qy0  = compute_error(this->qy0,  d_qy0_verified,  num_scale_coeffs);
		real error_z0   = compute_error(this->z0,   d_z0_verified,   num_scale_coeffs);

		free_device(d_eta0_verified);
		free_device(d_qx0_verified);
		free_device(d_qy0_verified);
		free_device(d_z0_verified);

		// mean
		// return (error_eta0 + error_qx0 + error_qy0 + error_z0) / C(4.0);
		
		// max
		return std::max({error_eta0, error_qx0, error_qy0, error_z0});
	}
	else if (this->solver_type == MWDG2)
	{
		char filename_eta0[255]  = {'\0'};
		char filename_qx0[255]   = {'\0'};
		char filename_qy0[255]   = {'\0'};
		char filename_z0[255]    = {'\0'};
		char filename_eta1x[255] = {'\0'};
		char filename_qx1x[255]  = {'\0'};
		char filename_qy1x[255]  = {'\0'};
		char filename_z1x[255]   = {'\0'};
		char filename_eta1y[255] = {'\0'};
		char filename_qx1y[255]  = {'\0'};
		char filename_qy1y[255]  = {'\0'};
		char filename_z1y[255]   = {'\0'};

		sprintf(filename_eta0,  "%s%s", prefix, "-scale-coeffs-eta0-mw");
		sprintf(filename_qx0,   "%s%s", prefix, "-scale-coeffs-qx0-mw");
		sprintf(filename_qy0,   "%s%s", prefix, "-scale-coeffs-qy0-mw");
		sprintf(filename_z0,    "%s%s", prefix, "-scale-coeffs-z0-mw");
		sprintf(filename_eta1x, "%s%s", prefix, "-scale-coeffs-eta1x-mw");
		sprintf(filename_qx1x,  "%s%s", prefix, "-scale-coeffs-qx1x-mw");
		sprintf(filename_qy1x,  "%s%s", prefix, "-scale-coeffs-qy1x-mw");
		sprintf(filename_z1x,   "%s%s", prefix, "-scale-coeffs-z1x-mw");
		sprintf(filename_eta1y, "%s%s", prefix, "-scale-coeffs-eta1y-mw");
		sprintf(filename_qx1y,  "%s%s", prefix, "-scale-coeffs-qx1y-mw");
		sprintf(filename_qy1y,  "%s%s", prefix, "-scale-coeffs-qy1y-mw");
		sprintf(filename_z1y,   "%s%s", prefix, "-scale-coeffs-z1y-mw");
		
		real* d_eta0_verified  = read_hierarchy_array_real(this->levels, dirroot, filename_eta0);
		real* d_qx0_verified   = read_hierarchy_array_real(this->levels, dirroot, filename_qx0);
		real* d_qy0_verified   = read_hierarchy_array_real(this->levels, dirroot, filename_qy0);
		real* d_z0_verified    = read_hierarchy_array_real(this->levels, dirroot, filename_z0);
		real* d_eta1x_verified = read_hierarchy_array_real(this->levels, dirroot, filename_eta1x);
		real* d_qx1x_verified  = read_hierarchy_array_real(this->levels, dirroot, filename_qx1x);
		real* d_qy1x_verified  = read_hierarchy_array_real(this->levels, dirroot, filename_qy1x);
		real* d_z1x_verified   = read_hierarchy_array_real(this->levels, dirroot, filename_z1x);
		real* d_eta1y_verified = read_hierarchy_array_real(this->levels, dirroot, filename_eta1y);
		real* d_qx1y_verified  = read_hierarchy_array_real(this->levels, dirroot, filename_qx1y);
		real* d_qy1y_verified  = read_hierarchy_array_real(this->levels, dirroot, filename_qy1y);
		real* d_z1y_verified   = read_hierarchy_array_real(this->levels, dirroot, filename_z1y);

		const int num_scale_coeffs = get_lvl_idx(this->levels + 1);

		const real error_eta0  = compute_error(this->eta0,  d_eta0_verified,  num_scale_coeffs);
		const real error_qx0   = compute_error(this->qx0,   d_qx0_verified,   num_scale_coeffs);
		const real error_qy0   = compute_error(this->qy0,   d_qy0_verified,   num_scale_coeffs);
		const real error_z0    = compute_error(this->z0,    d_z0_verified,    num_scale_coeffs);
		const real error_eta1x = compute_error(this->eta1x, d_eta1x_verified, num_scale_coeffs);
		const real error_qx1x  = compute_error(this->qx1x,  d_qx1x_verified,  num_scale_coeffs);
		const real error_qy1x  = compute_error(this->qy1x,  d_qy1x_verified,  num_scale_coeffs);
		const real error_z1x   = compute_error(this->z1x,   d_z1x_verified,   num_scale_coeffs);
		const real error_eta1y = compute_error(this->eta1y, d_eta1y_verified, num_scale_coeffs);
		const real error_qx1y  = compute_error(this->qx1y,  d_qx1y_verified,  num_scale_coeffs);
		const real error_qy1y  = compute_error(this->qy1y,  d_qy1y_verified,  num_scale_coeffs);
		const real error_z1y   = compute_error(this->z1y,   d_z1y_verified,   num_scale_coeffs);

		// mean
		// const real error_0  = (error_eta0  + error_qx0  + error_qy0  + error_z0)  / C(4.0);
		// const real error_1x = (error_eta1x + error_qx1x + error_qy1x + error_z1x) / C(4.0);
		// const real error_1y = (error_eta1y + error_qx1y + error_qy1y + error_z1y) / C(4.0);
		// const real error = (error_0 + error_1x + error_1y) / C(3.0);

		// max
		const real error = std::max({error_eta0,  error_qx0,  error_qy0,  error_z0,
                                     error_eta1x, error_qx1x, error_qy1x, error_z1x,
                                     error_eta1y, error_qx1y, error_qy1y, error_z1y});

		free_device(d_eta0_verified);
		free_device(d_qx0_verified);
		free_device(d_qy0_verified);
		free_device(d_z0_verified);
		free_device(d_eta1x_verified);
		free_device(d_qx1x_verified);
		free_device(d_qy1x_verified);
		free_device(d_z1x_verified);
		free_device(d_eta1y_verified);
		free_device(d_qx1y_verified);
		free_device(d_qy1y_verified);
		free_device(d_z1y_verified);

		return error;
	}
}
