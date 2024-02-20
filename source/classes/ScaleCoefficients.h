#pragma once

#include "../utilities/get_lvl_idx.cuh"
#include "../classes/SolverParams.h"
#include "../output/write_hierarchy_array_real.cuh"
#include "../input/read_hierarchy_array_real.cuh"
#include "../unittests/compute_error.cuh"

typedef struct ScaleCoefficients
{
	real* eta0 = nullptr;
	real* qx0  = nullptr;
	real* qy0  = nullptr;
	real* z0   = nullptr;

	real* eta1x = nullptr;
	real* qx1x  = nullptr;
	real* qy1x  = nullptr;
	real* z1x   = nullptr;

	real* eta1y = nullptr;
	real* qx1y  = nullptr;
	real* qy1y  = nullptr;
	real* z1y   = nullptr;

	bool is_copy_cuda = false;

	int levels = 0;
	int solver_type = 0;

	ScaleCoefficients
	(
		const SolverParams& solver_params
	)
		: levels(solver_params.L), solver_type(solver_params.solver_type)
	{
		const int num_all_elems = get_lvl_idx(this->levels + 1);
		
		const size_t bytes = sizeof(real) * num_all_elems;
		
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

	ScaleCoefficients
	(
		const SolverParams& solver_params,
		const char*         dirroot
	)
		: levels(solver_params.L), solver_type(solver_params.solver_type)
	{
		if (this->solver_type == HWFV1)
		{
			this->eta0 = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-eta0-hw");
			this->qx0  = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-qx0-hw");
			this->qy0  = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-qy0-hw");
			this->z0   = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-z0-hw");
		}
		else if (this->solver_type == MWDG2)
		{
			this->eta0  = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-eta0-mw");
			this->qx0   = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-qx0-mw");
			this->qy0   = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-qy0-mw");
			this->z0    = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-z0-mw");
			this->eta1x = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-eta1x-mw");
			this->qx1x  = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-qx1x-mw");
			this->qy1x  = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-qy1x-mw");
			this->z1x   = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-z1x-mw");
			this->eta1y = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-eta1y-mw");
			this->qx1y  = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-qx1y-mw");
			this->qy1y  = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-qy1y-mw");
			this->z1y   = read_hierarchy_array_real(this->levels, dirroot, "input-scale-coeffs-z1y-mw");
		}
	}

	ScaleCoefficients(const ScaleCoefficients& original) { *this = original; is_copy_cuda = true; }

	~ScaleCoefficients()
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

	void write_to_file
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

			sprintf(filename_eta0, "%s%c%s", prefix, '-', "scale-coeffs-eta0-hw");
			sprintf(filename_qx0,  "%s%c%s", prefix, '-', "scale-coeffs-qx0-hw");
			sprintf(filename_qy0,  "%s%c%s", prefix, '-', "scale-coeffs-qy0-hw");
			sprintf(filename_z0,   "%s%c%s", prefix, '-', "scale-coeffs-z0-hw");
			
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

			sprintf(filename_eta0,  "%s%c%s", prefix, '-', "scale-coeffs-eta0-mw");
			sprintf(filename_qx0,   "%s%c%s", prefix, '-', "scale-coeffs-qx0-mw");
			sprintf(filename_qy0,   "%s%c%s", prefix, '-', "scale-coeffs-qy0-mw");
			sprintf(filename_z0,    "%s%c%s", prefix, '-', "scale-coeffs-z0-mw");
			sprintf(filename_eta1x, "%s%c%s", prefix, '-', "scale-coeffs-eta1x-mw");
			sprintf(filename_qx1x,  "%s%c%s", prefix, '-', "scale-coeffs-qx1x-mw");
			sprintf(filename_qy1x,  "%s%c%s", prefix, '-', "scale-coeffs-qy1x-mw");
			sprintf(filename_z1x,   "%s%c%s", prefix, '-', "scale-coeffs-z1x-mw");
			sprintf(filename_eta1y, "%s%c%s", prefix, '-', "scale-coeffs-eta1y-mw");
			sprintf(filename_qx1y,  "%s%c%s", prefix, '-', "scale-coeffs-qx1y-mw");
			sprintf(filename_qy1y,  "%s%c%s", prefix, '-', "scale-coeffs-qy1y-mw");
			sprintf(filename_z1y,   "%s%c%s", prefix, '-', "scale-coeffs-z1y-mw");
			
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

	real verify
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

			sprintf(filename_eta0, "%s%c%s", prefix, '-', "scale-coeffs-eta0-hw");
			sprintf(filename_qx0,  "%s%c%s", prefix, '-', "scale-coeffs-qx0-hw");
			sprintf(filename_qy0,  "%s%c%s", prefix, '-', "scale-coeffs-qy0-hw");
			sprintf(filename_z0,   "%s%c%s", prefix, '-', "scale-coeffs-z0-hw");
			
			real* d_eta0_verified = read_hierarchy_array_real(this->levels, dirroot, filename_eta0);
			real* d_qx0_verified  = read_hierarchy_array_real(this->levels, dirroot, filename_qx0);
			real* d_qy0_verified  = read_hierarchy_array_real(this->levels, dirroot, filename_qy0);
			real* d_z0_verified   = read_hierarchy_array_real(this->levels, dirroot, filename_z0);

			const int num_scale_coeffs = get_lvl_idx(this->levels + 1);

			real error_eta0 = compute_error(dirroot, filename_eta0, this->eta0, d_eta0_verified, num_scale_coeffs);
			real error_qx0  = compute_error(dirroot, filename_qx0,  this->qx0,  d_qx0_verified,  num_scale_coeffs);
			real error_qy0  = compute_error(dirroot, filename_qy0,  this->qy0,  d_qy0_verified,  num_scale_coeffs);
			real error_z0   = compute_error(dirroot, filename_z0,   this->z0,   d_z0_verified,   num_scale_coeffs);

			free_device(d_eta0_verified);
			free_device(d_qx0_verified);
			free_device(d_qy0_verified);
			free_device(d_z0_verified);

			return (error_eta0 + error_qx0 + error_qy0 + error_z0) / C(4.0);
		}
	}

} ScaleCoefficients;