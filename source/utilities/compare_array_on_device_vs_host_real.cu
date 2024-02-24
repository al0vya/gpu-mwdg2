#include "compare_array_on_device_vs_host_real.cuh"

real compare_array_on_device_vs_host_real
(
	real*      h_array,
	real*      d_array,
	const int& array_length
)
{
	const size_t bytes = array_length * sizeof(real);
	real* h_array_copied = new real[array_length];

	copy_cuda(h_array_copied, d_array, bytes);

	real error      = C(0.0);
	real max_error  = C(0.0);
	real host_value = C(0.0);
	real file_value = C(0.0);

	for (int i = 0; i < array_length; i++)
	{
		host_value = h_array[i];

		file_value = h_array_copied[i];

		error = std::abs(host_value - file_value);

		max_error = std::max(max_error, error);
	}

	delete[] h_array_copied;

	return max_error;
}