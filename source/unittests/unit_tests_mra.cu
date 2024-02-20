#include "unit_tests_mra.cuh"

#if _RUN_UNIT_TESTS

#define TEST_MESSAGE_PASSED_ELSE_FAILED { printf("Passed unit test %s!\n", __func__); } else { printf("Failed unit test %s.\n", __func__); }

const real child_0 = C(1.0);
const real child_1 = C(2.0);
const real child_2 = C(3.0);
const real child_3 = C(4.0);

const ScaleChildrenHW s_HW = { child_0, child_1, child_2, child_3 };

void test_encode_scale()
{
	const real expected = C(0.5) * ( H0 * ( H0 * child_0 + H1 * child_2 ) + H1 * ( H0 * child_1 + H1 * child_3 ) );

	const real actual = encode_scale(s_HW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_alpha()
{
	const real expected  = C(0.5) * ( H0 * (G0 * child_0 + G1 * child_2) + H1 * (G0 * child_1 + G1 * child_3) );

	const real actual = encode_detail_alpha(s_HW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_beta()
{
	const real expected  = C(0.5) * ( G0 * (H0 * child_0 + H1 * child_2) + G1 * (H0 * child_1 + H1 * child_3) );

	const real actual = encode_detail_beta(s_HW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_gamma()
{
	const real expected = C(0.5) * ( G0 * (G0 * child_0 + G1 * child_2) + G1 * (G0 * child_1 + G1 * child_3) );

	const real actual = encode_detail_gamma(s_HW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

const real child_0_0  = C( 1.0);
const real child_0_1x = C( 2.0);
const real child_0_1y = C( 3.0);
const real child_1_0  = C( 4.0);
const real child_1_1x = C( 5.0);
const real child_1_1y = C( 6.0);
const real child_2_0  = C( 7.0);
const real child_2_1x = C( 8.0);
const real child_2_1y = C( 9.0);
const real child_3_0  = C(10.0);
const real child_3_1x = C(11.0);
const real child_3_1y = C(12.0);

const ScaleChildrenMW s_MW =
{
	{child_0_0,  child_1_0,  child_2_0,  child_3_0},
	{child_0_1x, child_1_1x, child_2_1x, child_3_1x},
	{child_0_1y, child_1_1y, child_2_1y, child_3_1y}
};

void test_encode_scale_0()
{
	const real expected = (HH0_11 * child_0_0 + HH0_12 * child_0_1x + HH0_13 * child_0_1y +
						   HH1_11 * child_2_0 + HH1_12 * child_2_1x + HH1_13 * child_2_1y +
						   HH2_11 * child_1_0 + HH2_12 * child_1_1x + HH2_13 * child_1_1y +
						   HH3_11 * child_3_0 + HH3_12 * child_3_1x + HH3_13 * child_3_1y) / C(2.0);

	const real actual = encode_scale_0(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_scale_1x()
{
	const real expected = (HH0_21 * child_0_0 + HH0_22 * child_0_1x + HH0_23 * child_0_1y +
						   HH1_21 * child_2_0 + HH1_22 * child_2_1x + HH1_23 * child_2_1y +
						   HH2_21 * child_1_0 + HH2_22 * child_1_1x + HH2_23 * child_1_1y +
						   HH3_21 * child_3_0 + HH3_22 * child_3_1x + HH3_23 * child_3_1y) / C(2.0);

	const real actual = encode_scale_1x(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_scale_1y()
{
	const real expected = (HH0_21 * child_0_0 + HH0_22 * child_0_1x + HH0_23 * child_0_1y +
						   HH1_21 * child_2_0 + HH1_22 * child_2_1x + HH1_23 * child_2_1y +
						   HH2_21 * child_1_0 + HH2_22 * child_1_1x + HH2_23 * child_1_1y +
						   HH3_21 * child_3_0 + HH3_22 * child_3_1x + HH3_23 * child_3_1y) / C(2.0);

	const real actual = encode_scale_1x(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_alpha_0()
{
	const real expected = (GA0_11 * child_0_0 + GA0_12 * child_0_1x + GA0_13 * child_0_1y +
						   GA1_11 * child_2_0 + GA1_12 * child_2_1x + GA1_13 * child_2_1y +
						   GA2_11 * child_1_0 + GA2_12 * child_1_1x + GA2_13 * child_1_1y +
						   GA3_11 * child_3_0 + GA3_12 * child_3_1x + GA3_13 * child_3_1y) / C(2.0);

	const real actual = encode_detail_alpha_0(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_beta_0()
{
	const real expected = (GB0_11 * child_0_0 + GB0_12 * child_0_1x + GB0_13 * child_0_1y +
						   GB1_11 * child_2_0 + GB1_12 * child_2_1x + GB1_13 * child_2_1y +
						   GB2_11 * child_1_0 + GB2_12 * child_1_1x + GB2_13 * child_1_1y +
						   GB3_11 * child_3_0 + GB3_12 * child_3_1x + GB3_13 * child_3_1y) / C(2.0);

	const real actual = encode_detail_beta_0(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_gamma_0()
{
	const real expected = (GC0_11 * child_0_0 + GC0_12 * child_0_1x + GC0_13 * child_0_1y +
						   GC1_11 * child_2_0 + GC1_12 * child_2_1x + GC1_13 * child_2_1y +
						   GC2_11 * child_1_0 + GC2_12 * child_1_1x + GC2_13 * child_1_1y +
						   GC3_11 * child_3_0 + GC3_12 * child_3_1x + GC3_13 * child_3_1y) / C(2.0);

	const real actual = encode_detail_gamma_0(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_alpha_1x()
{
	const real expected = (GA0_21 * child_0_0 + GA0_22 * child_0_1x + GA0_23 * child_0_1y +
						   GA1_21 * child_2_0 + GA1_22 * child_2_1x + GA1_23 * child_2_1y +
						   GA2_21 * child_1_0 + GA2_22 * child_1_1x + GA2_23 * child_1_1y +
						   GA3_21 * child_3_0 + GA3_22 * child_3_1x + GA3_23 * child_3_1y) / C(2.0);

	const real actual = encode_detail_alpha_1x(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_beta_1x()
{
	const real expected = (GB0_21 * child_0_0 + GB0_22 * child_0_1x + GB0_23 * child_0_1y +
						   GB1_21 * child_2_0 + GB1_22 * child_2_1x + GB1_23 * child_2_1y +
						   GB2_21 * child_1_0 + GB2_22 * child_1_1x + GB2_23 * child_1_1y +
						   GB3_21 * child_3_0 + GB3_22 * child_3_1x + GB3_23 * child_3_1y) / C(2.0);

	const real actual = encode_detail_beta_1x(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_gamma_1x()
{
	const real expected = (GC0_21 * child_0_0 + GC0_22 * child_0_1x + GC0_23 * child_0_1y +
						   GC1_21 * child_2_0 + GC1_22 * child_2_1x + GC1_23 * child_2_1y +
						   GC2_21 * child_1_0 + GC2_22 * child_1_1x + GC2_23 * child_1_1y +
						   GC3_21 * child_3_0 + GC3_22 * child_3_1x + GC3_23 * child_3_1y) / C(2.0);

	const real actual = encode_detail_gamma_1x(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_alpha_1y()
{
	const real expected = (GA0_31 * child_0_0 + GA0_32 * child_0_1x + GA0_33 * child_0_1y +
						   GA1_31 * child_2_0 + GA1_32 * child_2_1x + GA1_33 * child_2_1y +
						   GA2_31 * child_1_0 + GA2_32 * child_1_1x + GA2_33 * child_1_1y +
						   GA3_31 * child_3_0 + GA3_32 * child_3_1x + GA3_33 * child_3_1y) / C(2.0);

	const real actual = encode_detail_alpha_1y(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_beta_1y()
{
	const real expected = (GB0_31 * child_0_0 + GB0_32 * child_0_1x + GB0_33 * child_0_1y +
						   GB1_31 * child_2_0 + GB1_32 * child_2_1x + GB1_33 * child_2_1y +
						   GB2_31 * child_1_0 + GB2_32 * child_1_1x + GB2_33 * child_1_1y +
						   GB3_31 * child_3_0 + GB3_32 * child_3_1x + GB3_33 * child_3_1y) / C(2.0);

	const real actual = encode_detail_beta_1y(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void test_encode_detail_gamma_1y()
{
	const real expected = (GC0_31 * child_0_0 + GC0_32 * child_0_1x + GC0_33 * child_0_1y +
						   GC1_31 * child_2_0 + GC1_32 * child_2_1x + GC1_33 * child_2_1y +
						   GC2_31 * child_1_0 + GC2_32 * child_1_1x + GC2_33 * child_1_1y +
						   GC3_31 * child_3_0 + GC3_32 * child_3_1x + GC3_33 * child_3_1y) / C(2.0);

	const real actual = encode_detail_gamma_1y(s_MW);

	if ( are_reals_equal(actual, expected) )
		TEST_MESSAGE_PASSED_ELSE_FAILED
}

void unit_test_scale_coeffs()
{
	const char* dirroot = "unittestdata/unit_test_encode_and_thresh_topo";
	
	const SolverParams solver_params("unittestdata/unit_test_encode_and_thresh_topo/monai-hw.par");

	// L = 9 for Monai test case verification data
	ScaleCoefficients d_scale_coeffs(solver_params, dirroot);

	d_scale_coeffs.write_to_file(dirroot, "check");
}

void run_unit_tests_mra()
{
	test_encode_scale();
	test_encode_scale_0();
	test_encode_scale_1x();
	test_encode_scale_1y();

	test_encode_detail_alpha();
	test_encode_detail_beta();
	test_encode_detail_gamma();
	test_encode_detail_alpha_0();
	test_encode_detail_beta_0();
	test_encode_detail_gamma_0();
	test_encode_detail_alpha_1x();
	test_encode_detail_beta_1x();
	test_encode_detail_gamma_1x();
	test_encode_detail_alpha_1y();
	test_encode_detail_beta_1y();
	test_encode_detail_gamma_1y();

	unit_test_scale_coeffs();
}

#endif