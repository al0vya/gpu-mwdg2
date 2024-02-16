#pragma once

#include <cmath>
#include <cstdio>

#include "../mra/encode_and_thresh_topo.cuh"
#include "../utilities/are_reals_equal.h"

void test_encode_scale();
void test_encode_scale_0();
void test_encode_scale_1x();
void test_encode_scale_1y();

void test_encode_detail_alpha();
void test_encode_detail_beta();
void test_encode_detail_gamma();
void test_encode_detail_alpha_0();
void test_encode_detail_beta_0();
void test_encode_detail_gamma_0();
void test_encode_detail_alpha_1x();
void test_encode_detail_beta_1x();
void test_encode_detail_gamma_1x();
void test_encode_detail_alpha_1y();
void test_encode_detail_beta_1y();
void test_encode_detail_gamma_1y();

void unit_test_scale_coeffs();

void run_unit_tests_mra();