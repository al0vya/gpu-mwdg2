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

void run_unit_tests_mra()
{
	test_encode_scale();
	test_encode_scale_0();
	test_encode_scale_1x();
	test_encode_scale_1y();
}

#endif