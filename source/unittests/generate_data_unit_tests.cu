#include "generate_data_unit_tests.cuh"

void generate_data_unit_test_preflag_topo
(
	const char*       dirroot,
	const char*       input_or_output_str,
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
		input_or_output_str
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
		input_or_output_str,
		"-preflagged-details"
	);

	write_hierarchy_array_bool(dirroot, filename_preflagged_details, d_preflagged_details, solver_params.L - 1);
}

void generate_data_unit_test_encoding_all
(
	const char*       dirroot,
	const char*       input_or_output_str,
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	real*             d_norm_details,
	bool*             d_sig_details,
	bool*             d_preflagged_details,
	SolverParams      solver_params,
	const int&        timestep
)
{
	const std::string timestep_str = std::to_string(timestep);

	const std::string solver_str = (solver_params.solver_type == HWFV1) ? "_HW-" : "_MW-";
	
	const std::string prefix = "unit_test_encoding_all_TIMESTEP_" + timestep_str + solver_str + input_or_output_str;
	
	d_scale_coeffs.write_to_file( dirroot, prefix.c_str() );
	d_details.write_to_file( dirroot, prefix.c_str() );
	write_hierarchy_array_real(dirroot, (prefix + "-norm-details").c_str(),       d_norm_details,        solver_params.L - 1);
	write_hierarchy_array_bool(dirroot, (prefix + "-sig-details").c_str(),        d_sig_details,         solver_params.L - 1);
	write_hierarchy_array_bool(dirroot, (prefix + "-preflagged-details").c_str(), d_preflagged_details,  solver_params.L - 1);

}