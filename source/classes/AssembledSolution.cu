#include "AssembledSolution.h"

AssembledSolution::AssembledSolution
(
	const SolverParams& solver_params
)
:
	length( 1 << (2 * solver_params.L) ), max_length( 1 << (2 * solver_params.L) ), solver_type(solver_params.solver_type)
{
	this->initialise();
}

AssembledSolution::AssembledSolution
(
	const SolverParams& solver_params,
	const std::string&  name
)
:
	length( 1 << (2 * solver_params.L) ), max_length( 1 << (2 * solver_params.L) ), solver_type(solver_params.solver_type), name("-" + name)
{
	this->initialise();
}

AssembledSolution::AssembledSolution
(
	const SolverParams& solver_params,
	const char*         dirroot,
	const char*         prefix
)
:
	length( 1 << (2 * solver_params.L) ), max_length( 1 << (2 * solver_params.L) ), solver_type(solver_params.solver_type)
{
	this->initialise_from_file(dirroot, prefix);
}

AssembledSolution::AssembledSolution
(
	const SolverParams& solver_params,
	const std::string&  name,
	const char*         dirroot,
	const char*         prefix
)
:
	length( 1 << (2 * solver_params.L) ), max_length( 1 << (2 * solver_params.L) ), solver_type(solver_params.solver_type), name("-" + name)
{
	this->initialise_from_file(dirroot, prefix);
}

AssembledSolution::AssembledSolution
(
	const AssembledSolution& original
)
{
	*this = original;
	 this->is_copy_cuda = true;
}

AssembledSolution::~AssembledSolution()
{
	if (!is_copy_cuda)
	{
		CHECK_CUDA_ERROR( free_device(h0) );
		CHECK_CUDA_ERROR( free_device(qx0) );
		CHECK_CUDA_ERROR( free_device(qy0) );
		CHECK_CUDA_ERROR( free_device(z0) );

		CHECK_CUDA_ERROR( free_device(h1x) );
		CHECK_CUDA_ERROR( free_device(qx1x) );
		CHECK_CUDA_ERROR( free_device(qy1x) );
		CHECK_CUDA_ERROR( free_device(z1x) );

		CHECK_CUDA_ERROR( free_device(h1y) );
		CHECK_CUDA_ERROR( free_device(qx1y) );
		CHECK_CUDA_ERROR( free_device(qy1y) );
		CHECK_CUDA_ERROR( free_device(z1y) );

		CHECK_CUDA_ERROR( free_device(act_idcs) );
		CHECK_CUDA_ERROR( free_device(levels) );
	}
}

void AssembledSolution::write_to_file
(
	const char* dirroot,
	const char* prefix
)
{
	if (this->solver_type == HWFV1)
	{
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-h0-hw" ).c_str(),  this->h0,  this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-qx0-hw" ).c_str(), this->qx0, this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-qy0-hw" ).c_str(), this->qy0, this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-z0-hw" ).c_str(),  this->z0,  this->max_length);

		write_d_array_int(dirroot, ( std::string(prefix) + this->name + "-assem-sol-levels-hw" ).c_str(),   this->levels,   this->max_length);
		write_d_array_int(dirroot, ( std::string(prefix) + this->name + "-assem-sol-act-idcs-hw" ).c_str(), this->act_idcs, this->max_length);
	}
	else if (this->solver_type == MWDG2)
	{
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-h0-mw" ).c_str(),  this->h0,  this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-qx0-mw" ).c_str(), this->qx0, this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-qy0-mw" ).c_str(), this->qy0, this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-z0-mw" ).c_str(),  this->z0,  this->max_length);
		
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-h1x-mw" ).c_str(),  this->h1x,  this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-qx1x-mw" ).c_str(), this->qx1x, this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-qy1x-mw" ).c_str(), this->qy1x, this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-z1x-mw" ).c_str(),  this->z1x,  this->max_length);
		
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-h1y-mw" ).c_str(),  this->h1y,  this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-qx1y-mw" ).c_str(), this->qx1y, this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-qy1y-mw" ).c_str(), this->qy1y, this->max_length);
		write_d_array_real(dirroot, ( std::string(prefix) + this->name + "-assem-sol-z1y-mw" ).c_str(),  this->z1y,  this->max_length);

		write_d_array_int(dirroot, ( std::string(prefix) + this->name + "-assem-sol-levels-mw" ).c_str(),   this->levels,   this->max_length);
		write_d_array_int(dirroot, ( std::string(prefix) + this->name + "-assem-sol-act-idcs-mw" ).c_str(), this->act_idcs, this->max_length);
	}
}

real AssembledSolution::verify_real
(
	const char* dirroot,
	const char* prefix
)
{
	if (this->solver_type == HWFV1)
	{
		real* d_h0_verified  = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-h0-hw" ).c_str() );
		real* d_qx0_verified = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qx0-hw" ).c_str() );
		real* d_qy0_verified = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qy0-hw" ).c_str() );
		real* d_z0_verified  = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-z0-hw" ).c_str() );

		real error_h0  = compute_max_error(this->h0,  d_h0_verified,  this->length);
		real error_qx0 = compute_max_error(this->qx0, d_qx0_verified, this->length);
		real error_qy0 = compute_max_error(this->qy0, d_qy0_verified, this->length);
		real error_z0  = compute_max_error(this->z0,  d_z0_verified,  this->length);

		free_device(d_h0_verified);
		free_device(d_qx0_verified);
		free_device(d_qy0_verified);
		free_device(d_z0_verified);

		// mean
		// return (error_h0 + error_qx0 + error_qy0 + error_z0) / C(4.0);

		// max
		return std::max({ error_h0, error_qx0, error_qy0, error_z0 });
	}
	else if (this->solver_type == MWDG2)
	{
		real* d_h0_verified   = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-h0-mw" ).c_str() );
		real* d_qx0_verified  = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qx0-mw" ).c_str() );
		real* d_qy0_verified  = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qy0-mw" ).c_str() );
		real* d_z0_verified   = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-z0-mw" ).c_str() );
		real* d_h1x_verified  = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-h1x-mw" ).c_str() );
		real* d_qx1x_verified = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qx1x-mw" ).c_str() );
		real* d_qy1x_verified = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qy1x-mw" ).c_str() );
		real* d_z1x_verified  = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-z1x-mw" ).c_str() );
		real* d_h1y_verified  = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-h1y-mw" ).c_str() );
		real* d_qx1y_verified = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qx1y-mw" ).c_str() );
		real* d_qy1y_verified = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qy1y-mw" ).c_str() );
		real* d_z1y_verified  = read_d_array_real( this->length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-z1y-mw" ).c_str() );
		
		real error_h0   = compute_max_error(this->h0,   d_h0_verified,   this->length);
		real error_qx0  = compute_max_error(this->qx0,  d_qx0_verified,  this->length);
		real error_qy0  = compute_max_error(this->qy0,  d_qy0_verified,  this->length);
		real error_z0   = compute_max_error(this->z0,   d_z0_verified,   this->length);
		real error_h1x  = compute_max_error(this->h1x,  d_h1x_verified,  this->length);
		real error_qx1x = compute_max_error(this->qx1x, d_qx1x_verified, this->length);
		real error_qy1x = compute_max_error(this->qy1x, d_qy1x_verified, this->length);
		real error_z1x  = compute_max_error(this->z1x,  d_z1x_verified,  this->length);
		real error_h1y  = compute_max_error(this->h1y,  d_h1y_verified,  this->length);
		real error_qx1y = compute_max_error(this->qx1y, d_qx1y_verified, this->length);
		real error_qy1y = compute_max_error(this->qy1y, d_qy1y_verified, this->length);
		real error_z1y  = compute_max_error(this->z1y,  d_z1y_verified,  this->length);

		free_device(d_h0_verified);
		free_device(d_qx0_verified);
		free_device(d_qy0_verified);
		free_device(d_z0_verified);
		free_device(d_h1x_verified);
		free_device(d_qx1x_verified);
		free_device(d_qy1x_verified);
		free_device(d_z1x_verified);
		free_device(d_h1y_verified);
		free_device(d_qx1y_verified);
		free_device(d_qy1y_verified);
		free_device(d_z1y_verified);

		// mean
		// const real error_0  = (error_eta0  + error_qx0  + error_qy0  + error_z0)  / C(4.0);
		// const real error_1x = (error_eta1x + error_qx1x + error_qy1x + error_z1x) / C(4.0);
		// const real error_1y = (error_eta1y + error_qx1y + error_qy1y + error_z1y) / C(4.0);
		// const real error = (error_0 + error_1x + error_1y) / C(3.0);

		// max
		const real error = std::max({error_h0,  error_qx0,  error_qy0,  error_z0,
                                     error_h1x, error_qx1x, error_qy1x, error_z1x,
                                     error_h1y, error_qx1y, error_qy1y, error_z1y});

		return error;
	}
}

int AssembledSolution::verify_int
(
	const char* dirroot,
	const char* prefix
)
{
	if (this->solver_type == HWFV1)
	{
		int diff_levels   = compare_d_array_with_file_int(dirroot, ( std::string(prefix) + this->name + "-assem-sol-levels-hw"   ).c_str(), this->levels,   this->length, 0);
		int diff_act_idcs = compare_d_array_with_file_int(dirroot, ( std::string(prefix) + this->name + "-assem-sol-act-idcs-hw" ).c_str(), this->act_idcs, this->length, 0);

		return diff_levels + diff_act_idcs;
	}
	else if (this->solver_type == MWDG2)
	{
		int diff_levels   = compare_d_array_with_file_int(dirroot, ( std::string(prefix) + this->name + "-assem-sol-levels-mw"   ).c_str(), this->levels,   this->length, 0);
		int diff_act_idcs = compare_d_array_with_file_int(dirroot, ( std::string(prefix) + this->name + "-assem-sol-act-idcs-mw" ).c_str(), this->act_idcs, this->length, 0);

		return diff_levels + diff_act_idcs;
	}
}

void AssembledSolution::initialise()
{
	size_t bytes_real = this->max_length * sizeof(real);
	size_t bytes_int  = this->max_length * sizeof(HierarchyIndex);

	this->h0  = (real*)malloc_device(bytes_real);
	this->qx0 = (real*)malloc_device(bytes_real);
	this->qy0 = (real*)malloc_device(bytes_real);
	this->z0  = (real*)malloc_device(bytes_real);

	this->act_idcs = (HierarchyIndex*)malloc_device(bytes_int);
	this->levels   = (int*)malloc_device(bytes_int);

	const int num_blocks = get_num_blocks(this->max_length, THREADS_PER_BLOCK);

	zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->h0,       this->max_length);
	zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->qx0,      this->max_length);
	zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->qy0,      this->max_length);
	zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->z0,       this->max_length);
	zero_array_kernel_int<<<num_blocks,  THREADS_PER_BLOCK>>>(this->levels,   this->max_length);
	zero_array_kernel_int<<<num_blocks,  THREADS_PER_BLOCK>>>(this->act_idcs, this->max_length);

	if (this->solver_type == MWDG2)
	{
		this->h1x  = (real*)malloc_device(bytes_real);
		this->qx1x = (real*)malloc_device(bytes_real);
		this->qy1x = (real*)malloc_device(bytes_real);
		this->z1x  = (real*)malloc_device(bytes_real);

		this->h1y  = (real*)malloc_device(bytes_real);
		this->qx1y = (real*)malloc_device(bytes_real);
		this->qy1y = (real*)malloc_device(bytes_real);
		this->z1y  = (real*)malloc_device(bytes_real);

		zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->h1x,  this->max_length);
		zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->qx1x, this->max_length);
		zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->qy1x, this->max_length);
		zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->z1x,  this->max_length);

		zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->h1y,  this->max_length);
		zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->qx1y, this->max_length);
		zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->qy1y, this->max_length);
		zero_array_kernel_real<<<num_blocks, THREADS_PER_BLOCK>>>(this->z1y,  this->max_length);
	}
}

void AssembledSolution::initialise_from_file
(
	const char* dirroot,
	const char* prefix
)
{
	if (this->solver_type == HWFV1)
	{
		this->h0  = read_d_array_real( this->max_length, dirroot, (std::string(prefix) + this->name + "-assem-sol-h0-hw" ).c_str() );
		this->qx0 = read_d_array_real( this->max_length, dirroot, (std::string(prefix) + this->name + "-assem-sol-qx0-hw").c_str() );
		this->qy0 = read_d_array_real( this->max_length, dirroot, (std::string(prefix) + this->name + "-assem-sol-qy0-hw").c_str() );
		this->z0  = read_d_array_real( this->max_length, dirroot, (std::string(prefix) + this->name + "-assem-sol-z0-hw" ).c_str() );

		this->levels   = read_d_array_int( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-levels-hw"   ).c_str() );
		this->act_idcs = read_d_array_int( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-act-idcs-hw" ).c_str() );
	}
	else if (this->solver_type == MWDG2)
	{
		this->h0  = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-h0-mw"  ).c_str() );
		this->qx0 = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qx0-mw" ).c_str() );
		this->qy0 = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qy0-mw" ).c_str() );
		this->z0  = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-z0-mw"  ).c_str() );
		
		this->h1x  = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-h1x-mw"  ).c_str() );
		this->qx1x = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qx1x-mw" ).c_str() );
		this->qy1x = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qy1x-mw" ).c_str() );
		this->z1x  = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-z1x-mw"  ).c_str() );
		
		this->h1y  = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-h1y-mw"  ).c_str() );
		this->qx1y = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qx1y-mw" ).c_str() );
		this->qy1y = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-qy1y-mw" ).c_str() );
		this->z1y  = read_d_array_real( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-z1y-mw"  ).c_str() );

		this->levels   = read_d_array_int( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-levels-mw"   ).c_str() );
		this->act_idcs = read_d_array_int( this->max_length, dirroot, ( std::string(prefix) + this->name + "-assem-sol-act-idcs-mw" ).c_str() );
	}
}