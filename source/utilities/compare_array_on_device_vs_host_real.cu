#include "compare_array_on_device_vs_host_real.cuh"

bool compare_array_on_device_vs_host_real
(
	real*      h_array,
	real*      d_array,
	const int& array_length
)
{
	const size_t bytes = array_length * sizeof(real);
	real* h_array_copied = new real[array_length];

	copy_cuda(h_array_copied, d_array, bytes);

	bool passed = true;

	for (int i = 0; i < array_length; i++)
	{
		if ( !are_reals_equal( h_array[i], h_array_copied[i] ) )
		{
			passed = false;
			break;
		}
	}

	delete[] h_array_copied;

	return passed;
}