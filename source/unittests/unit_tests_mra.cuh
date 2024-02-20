#pragma once

#include <cmath>
#include <cstdio>

#include "../mra/preflag_topo.cuh"
#include "../utilities/are_reals_equal.h"
#include "../input/read_hierarchy_array_bool.cuh"
#include "../output/write_hierarchy_array_bool.cuh"

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

void unit_test_preflag_topo_hw();

void run_unit_tests_mra();