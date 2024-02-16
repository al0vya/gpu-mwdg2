#pragma once

#include "../utilities/cuda_utils.cuh"
#include "../utilities/get_lvl_idx.cuh"

#include "../types/real.h"
#include "../types/SolverTypes.h"


#include "../output/write_hierarchy_to_file.cuh"

typedef struct ScaleCoefficients
{
	real* eta0;
	real* qx0;
	real* qy0;
	real* z0;

	real* eta1x;
	real* qx1x;
	real* qy1x;
	real* z1x;

	real* eta1y;
	real* qx1y;
	real* qy1y;
	real* z1y;

	bool is_copy_cuda = false;

	int solver_type;

	ScaleCoefficients
	(
		const int& num_all_elems, 
		const int& solver_type
	)
		: solver_type(solver_type)
	{
		size_t bytes = sizeof(real) * num_all_elems;
		
		eta0 = (real*)malloc_device(bytes);
		qx0  = (real*)malloc_device(bytes);
		qy0  = (real*)malloc_device(bytes);
		z0   = (real*)malloc_device(bytes);

		eta1x = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		qx1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		qy1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		z1x   = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;

		eta1y = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		qx1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		qy1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		z1y   = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
	}

	// constructor based on reading files from
	// Monai test case, with L = 9, for unit testing
	ScaleCoefficients
	(
		const int& solver_type
	)
		: solver_type(solver_type)
	{
		const int num_all_elems = get_lvl_idx(9+1); // L = 9 then + 1

		size_t bytes = sizeof(real) * num_all_elems;

		// ALLOCATING DEVICE BUFFERS //
		
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
		
		// ------------------------- //
		
		
		// ALLOCATING HOST BUFFERS //
		
		real* h_eta0 = new real[num_all_elems];
		real* h_qx0  = new real[num_all_elems];
		real* h_qy0  = new real[num_all_elems];
		real* h_z0   = new real[num_all_elems];
		
		real* h_eta1x = (solver_type == MWDG2) ? new real[num_all_elems] : nullptr;
		real* h_qx1x  = (solver_type == MWDG2) ? new real[num_all_elems] : nullptr;
		real* h_qy1x  = (solver_type == MWDG2) ? new real[num_all_elems] : nullptr;
		real* h_z1x   = (solver_type == MWDG2) ? new real[num_all_elems] : nullptr;
		
		real* h_eta1y = (solver_type == MWDG2) ? new real[num_all_elems] : nullptr;
		real* h_qx1y  = (solver_type == MWDG2) ? new real[num_all_elems] : nullptr;
		real* h_qy1y  = (solver_type == MWDG2) ? new real[num_all_elems] : nullptr;
		real* h_z1y   = (solver_type == MWDG2) ? new real[num_all_elems] : nullptr;

		// ------------------------//

		if (solver_type == HWFV1) // reading from file for GPU-HWFV1
		{
			FILE* fp_eta0 = fopen("unittestdata/unit_test_encode_and_thresh_topo/scale-coeffs-eta0-hw.txt", "r");
			FILE* fp_qx0  = fopen("unittestdata/unit_test_encode_and_thresh_topo/scale-coeffs-qx0-hw.txt",  "r");
			FILE* fp_qy0  = fopen("unittestdata/unit_test_encode_and_thresh_topo/scale-coeffs-qy0-hw.txt",  "r");
			FILE* fp_z0   = fopen("unittestdata/unit_test_encode_and_thresh_topo/scale-coeffs-z0-hw.txt",   "r");

			real eta0_dummy = C(0.0);
			real qx0_dummy  = C(0.0);
			real qy0_dummy  = C(0.0);
			real z0_dummy   = C(0.0);

			for (int i = 0; i < num_all_elems; i++) // reading from file for GPU-MWDG2
			{
				fscanf(fp_eta0, "%" NUM_FRMT, &eta0_dummy);
				fscanf(fp_qx0,  "%" NUM_FRMT, &qx0_dummy);
				fscanf(fp_qy0,  "%" NUM_FRMT, &qy0_dummy);
				fscanf(fp_z0,   "%" NUM_FRMT, &z0_dummy);

				h_eta0[i] = eta0_dummy;
				h_qx0[i]  = qx0_dummy;
				h_qy0[i]  = qy0_dummy;
				h_z0[i]   = z0_dummy;
			}

			copy_cuda(this->eta0, h_eta0, bytes);
			copy_cuda(this->qx0,  h_qx0,  bytes);
			copy_cuda(this->qy0,  h_qy0,  bytes);
			copy_cuda(this->z0,   h_z0,   bytes);

			fclose(fp_eta0);
			fclose(fp_qx0);
			fclose(fp_qy0);
			fclose(fp_z0);
		}
		else if (solver_type == MWDG2)
		{

		}

		delete[] h_eta0;
		delete[] h_qx0;
		delete[] h_qy0;
		delete[] h_z0;

		if (solver_type == MWDG2)
		{
			delete[] h_eta1x;
			delete[] h_qx1x;
			delete[] h_qy1x;
			delete[] h_z1x;

			delete[] h_eta1y;
			delete[] h_qx1y;
			delete[] h_qy1y;
			delete[] h_z1y;
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

	void write_to_file()
	{
		if (this->solver_type == HWFV1)
		{
			write_hierarchy_to_file("unittestdata/unit_test_encode_and_thresh_topo/output-scale-coeffs-eta0-hw", this->eta0, 9);
			write_hierarchy_to_file("unittestdata/unit_test_encode_and_thresh_topo/output-scale-coeffs-qx0-hw",  this->qx0,  9);
			write_hierarchy_to_file("unittestdata/unit_test_encode_and_thresh_topo/output-scale-coeffs-qy0-hw",  this->qy0,  9);
			write_hierarchy_to_file("unittestdata/unit_test_encode_and_thresh_topo/output-scale-coeffs-z0-hw",   this->z0,   9);
		}
	}

} ScaleCoefficients;