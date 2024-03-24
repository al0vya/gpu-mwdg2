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

void generate_data_unit_test_encode_flow
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
	
	const std::string prefix = "unit_test_encode_flow_TIMESTEP_" + timestep_str + solver_str + input_or_output_str;
	
	d_scale_coeffs.write_to_file( dirroot, prefix.c_str() );
	d_details.write_to_file( dirroot, prefix.c_str() );
	write_hierarchy_array_real(dirroot, (prefix + "-norm-details").c_str(),       d_norm_details,        solver_params.L - 1);
	write_hierarchy_array_bool(dirroot, (prefix + "-sig-details").c_str(),        d_sig_details,         solver_params.L - 1);
	write_hierarchy_array_bool(dirroot, (prefix + "-preflagged-details").c_str(), d_preflagged_details,  solver_params.L - 1);

}

void generate_data_unit_test_decoding
(
	const char*       dirroot,
	const char*       input_or_output_str,
	bool*             d_sig_details,
	real*             d_norm_details,
	Details           d_details,
	ScaleCoefficients d_scale_coeffs,
	SolverParams      solver_params,	
	const int&        timestep
)
{
	const std::string timestep_str = std::to_string(timestep);

	const std::string solver_str = (solver_params.solver_type == HWFV1) ? "_HW-" : "_MW-";
	
	const std::string prefix = "unit_test_decoding_TIMESTEP_" + timestep_str + solver_str + input_or_output_str;
	
	d_scale_coeffs.write_to_file( dirroot, prefix.c_str() );
	d_details.write_to_file( dirroot, prefix.c_str() );
	write_hierarchy_array_real(dirroot, (prefix + "-norm-details").c_str(), d_norm_details, solver_params.L - 1);
	write_hierarchy_array_bool(dirroot, (prefix + "-sig-details").c_str(),  d_sig_details,  solver_params.L - 1);
}

void generate_data_unit_test_regularisation
(
	const char*  dirroot,
	const char*  input_or_output_str,
	bool*        d_sig_details,
	SolverParams solver_params,	
	const int&   timestep
)
{
	const std::string timestep_str = std::to_string(timestep);

	const std::string solver_str = (solver_params.solver_type == HWFV1) ? "_HW-" : "_MW-";
	
	const std::string prefix = "unit_test_regularisation_TIMESTEP_" + timestep_str + solver_str + input_or_output_str;
	
	write_hierarchy_array_bool(dirroot, (prefix + "-sig-details").c_str(), d_sig_details, solver_params.L - 1);
}

void generate_data_unit_test_extra_significance
(
	const char*  dirroot,
	const char*  input_or_output_str,
	bool*        d_sig_details,
	real*        d_norm_details,
	SolverParams solver_params,	
	const int&   timestep
)
{
	const std::string timestep_str = std::to_string(timestep);

	const std::string solver_str = (solver_params.solver_type == HWFV1) ? "_HW-" : "_MW-";
	
	const std::string prefix = "unit_test_extra_significance_TIMESTEP_" + timestep_str + solver_str + input_or_output_str;
	
	write_hierarchy_array_bool(dirroot, (prefix + "-sig-details").c_str(), d_sig_details, solver_params.L - 1);
	write_hierarchy_array_real(dirroot, (prefix + "-norm-details").c_str(), d_norm_details, solver_params.L - 1);
}

void generate_data_unit_test_dg2_update_RK1
(
	const char*        dirroot,
	const char*        input_or_output_str,
	Neighbours&        d_neighbours,
	AssembledSolution& d_assem_sol,
	AssembledSolution& d_buf_assem_sol,
	const real&        dx_finest,
	const real&        dy_finest,
	const real&        dt,
	real*              d_dt_CFL,
	const int&         timestep
)
{
	const std::string timestep_str = std::to_string(timestep);
	const std::string prefix       = "unit_test_dg2_update_RK1_TIMESTEP_" + timestep_str + "-" + input_or_output_str;

	d_neighbours.write_to_file( dirroot, prefix.c_str() );
	d_assem_sol.write_to_file( dirroot, prefix.c_str() );
	d_buf_assem_sol.write_to_file( dirroot, prefix.c_str() );
	write_d_array_real(dirroot, (prefix + "-dt-CFL").c_str(), d_dt_CFL, d_assem_sol.max_length);

	printf("Variable data for %s:\n", prefix.c_str());
	printf("dx_finest:          %f\n", dx_finest);
	printf("dy_finest:          %f\n", dy_finest);
	printf("dt:                 %f\n", dt);
	printf("d_assem_sol.length: %d\n", d_assem_sol.length);
}

void generate_data_unit_test_dg2_update_RK2
(
	const char*        dirroot,
	const char*        input_or_output_str,
	Neighbours&        d_neighbours,
	AssembledSolution& d_buf_assem_sol,
	AssembledSolution& d_assem_sol,
	const real&        dx_finest,
	const real&        dy_finest,
	const real&        dt,
	real*              d_dt_CFL,
	const int&         timestep
)
{
	const std::string timestep_str = std::to_string(timestep);
	const std::string prefix       = "unit_test_dg2_update_RK2_TIMESTEP_" + timestep_str + "-" + input_or_output_str;

	d_neighbours.write_to_file( dirroot, prefix.c_str() );
	d_buf_assem_sol.write_to_file( dirroot, prefix.c_str() );
	d_assem_sol.write_to_file( dirroot, prefix.c_str() );
	write_d_array_real(dirroot, (prefix + "-dt-CFL").c_str(), d_dt_CFL, d_assem_sol.max_length);

	printf("Variable data for %s:\n", prefix.c_str());
	printf("dx_finest:          %f\n", dx_finest);
	printf("dy_finest:          %f\n", dy_finest);
	printf("dt:                 %f\n", dt);
	printf("d_assem_sol.length: %d\n", d_assem_sol.length);
}