#pragma once

#include <cmath>
#include <cstdio>
#include <string>

#include "../types/real.h"
#include "../mra/preflag_topo.cuh"
#include "../mra/encoding_all.cuh"
#include "../utilities/are_reals_equal.h"
#include "../input/read_hierarchy_array_bool.cuh"
#include "../output/write_hierarchy_array_bool.cuh"

void unit_test_encode_scale();
void unit_test_encode_scale_0();
void unit_test_encode_scale_1x();
void unit_test_encode_scale_1y();

void unit_test_encode_detail_alpha();
void unit_test_encode_detail_beta();
void unit_test_encode_detail_gamma();
void unit_test_encode_detail_alpha_0();
void unit_test_encode_detail_beta_0();
void unit_test_encode_detail_gamma_0();
void unit_test_encode_detail_alpha_1x();
void unit_test_encode_detail_beta_1x();
void unit_test_encode_detail_gamma_1x();
void unit_test_encode_detail_alpha_1y();
void unit_test_encode_detail_beta_1y();
void unit_test_encode_detail_gamma_1y();

void unit_test_preflag_topo_HW();

void unit_test_encoding_all_TIMESTEP_1_HW();

void run_unit_tests_mra();