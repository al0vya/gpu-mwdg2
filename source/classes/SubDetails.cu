#include "SubDetails.h"

SubDetails::SubDetails() = default;

SubDetails::SubDetails
(
	const int& levels
)
	: levels(levels)
{
	if (this->levels != -1)
	{
		const int num_details = get_lvl_idx(this->levels + 1);
		const int num_blocks  = get_num_blocks(num_details, THREADS_PER_BLOCK);
		size_t    bytes       = sizeof(real) * num_details;

		this->alpha = (real*)malloc_device(bytes);
		this->beta  = (real*)malloc_device(bytes);
		this->gamma = (real*)malloc_device(bytes);

		zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(this->alpha, num_details);
		zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(this->beta,  num_details);
		zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(this->gamma, num_details);
	}
}

SubDetails::SubDetails
(
	const int&  levels,
	const char* dirroot,
	const char* prefix,
	const char* suffix
)
	: levels(levels)
{
	const int num_details = get_lvl_idx(this->levels + 1);
	
	char filename_alpha[255] = {'\0'};
	char filename_beta [255] = {'\0'};
	char filename_gamma[255] = {'\0'};

	sprintf(filename_alpha, "%s%s%s", prefix, "-details-alpha-", suffix);
	sprintf(filename_beta , "%s%s%s", prefix, "-details-beta-",  suffix);
	sprintf(filename_gamma, "%s%s%s", prefix, "-details-gamma-", suffix);

	alpha = (this->levels > -1) ? read_hierarchy_array_real(this->levels, dirroot, filename_alpha) : nullptr;
	beta  = (this->levels > -1) ? read_hierarchy_array_real(this->levels, dirroot, filename_beta)  : nullptr;
	gamma = (this->levels > -1) ? read_hierarchy_array_real(this->levels, dirroot, filename_gamma) : nullptr;
}

SubDetails::SubDetails(const SubDetails& original) { *this = original; is_copy_cuda = true; }

SubDetails::~SubDetails()
{
	if (!is_copy_cuda)
	{
		CHECK_CUDA_ERROR( free_device(alpha) );
		CHECK_CUDA_ERROR( free_device(beta) );
		CHECK_CUDA_ERROR( free_device(gamma) );
	}
}

void SubDetails::write_to_file
(
	const char* dirroot,
	const char* prefix,
	const char* suffix
)
{
	char filename_alpha[255] = {'\0'};
	char filename_beta [255] = {'\0'};
	char filename_gamma[255] = {'\0'};

	sprintf(filename_alpha, "%s%s%s", prefix, "-details-alpha-", suffix);
	sprintf(filename_beta , "%s%s%s", prefix, "-details-beta-",  suffix);
	sprintf(filename_gamma, "%s%s%s", prefix, "-details-gamma-", suffix);

	write_hierarchy_array_real(dirroot, filename_alpha, this->alpha, this->levels);
	write_hierarchy_array_real(dirroot, filename_beta,  this->beta,  this->levels);
	write_hierarchy_array_real(dirroot, filename_gamma, this->gamma, this->levels);
}

real SubDetails::verify
(
	const char* dirroot,
	const char* prefix,
	const char* suffix
)
{
	char filename_alpha[255] = {'\0'};
	char filename_beta [255] = {'\0'};
	char filename_gamma[255] = {'\0'};

	sprintf(filename_alpha, "%s%c%s%c%s", prefix, '-', "details-alpha", '-', suffix);
	sprintf(filename_beta , "%s%c%s%c%s", prefix, '-', "details-beta",  '-', suffix);
	sprintf(filename_gamma, "%s%c%s%c%s", prefix, '-', "details-gamma", '-', suffix);

	real* d_alpha = read_hierarchy_array_real(this->levels, dirroot, filename_alpha);
	real* d_beta  = read_hierarchy_array_real(this->levels, dirroot, filename_beta);
	real* d_gamma = read_hierarchy_array_real(this->levels, dirroot, filename_gamma);

	const int num_details = get_lvl_idx(this->levels + 1);

	real error_alpha = compute_max_error(this->alpha, d_alpha, num_details);
	real error_beta  = compute_max_error(this->beta , d_beta , num_details);
	real error_gamma = compute_max_error(this->gamma, d_gamma, num_details);

	free_device(d_alpha);
	free_device(d_beta );
	free_device(d_gamma);

	// mean
	// return (error_alpha + error_beta + error_gamma) / C(3.0);

	// max
	return max( error_alpha, max(error_beta, error_gamma) );
}