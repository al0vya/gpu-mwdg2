#include "generate_data_unit_tests.cuh"

void generate_data_unit_test_preflag_topo
(
	const char*       dirroot,
	const char*       input_or_output_message,
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	bool*             d_preflagged_details,
	SolverParams      solver_params
)
{
	char prefix[255] = {'\0'};

	sprintf
	(
		prefix,
		"%s%s%s",
		"unit_test_preflag_topo_",
		(solver_params.solver_type == HWFV1) ? "HW-" : "MW-",
		input_or_output_message
	);

	d_scale_coeffs.write_to_file(dirroot, prefix);
	d_details.write_to_file(dirroot, prefix);

	char filename_preflagged_details[255] = {'\0'};

	sprintf
	(
		filename_preflagged_details,
		"%s%s%s%s",
		"unit_test_preflag_topo_",
		(d_scale_coeffs.solver_type == HWFV1) ? "HW-" : "MW-",
		input_or_output_message,
		"-preflagged-details"
	);

	write_hierarchy_array_bool(dirroot, filename_preflagged_details, d_preflagged_details, solver_params.L - 1);
}