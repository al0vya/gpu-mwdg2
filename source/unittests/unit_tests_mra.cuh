#pragma once

#include <cmath>
#include <cstdio>
#include <string>

#include "../types/real.h"
#include "../mra/preflag_topo.cuh"
#include "../mra/encode_flow.cuh"
#include "../mra/decoding.cuh"
#include "../utilities/are_reals_equal.h"
#include "../utilities/compare_d_array_with_file_bool.cuh"
#include "../utilities/compare_d_array_with_file_real.cuh"
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
void unit_test_encode_flow_TIMESTEP_1_HW();
void unit_test_encode_flow_TIMESTEP_2_HW();
void unit_test_decoding_TIMESTEP_1_HW();
void unit_test_decoding_TIMESTEP_2_HW();

void unit_test_preflag_topo_MW();
void unit_test_encode_flow_TIMESTEP_1_MW();
void unit_test_encode_flow_TIMESTEP_2_MW();
void unit_test_decoding_TIMESTEP_1_MW();
void unit_test_decoding_TIMESTEP_2_MW();

void run_unit_tests_mra();