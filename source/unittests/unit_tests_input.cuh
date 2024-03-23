#pragma once

#include "../input/read_keyword_int.h"
#include "../input/read_keyword_str.h"
#include "../input/read_cell_size.h"
#include "../input/read_d_array_int.cuh"
#include "../input/read_d_array_real.cuh"
#include "../input/read_hierarchy_array_real.cuh"
#include "../input/read_hierarchy_array_bool.cuh"
#include "../input/read_hierarchy_array_bool.cuh"
#include "../utilities/are_reals_equal.h"

void unit_test_read_keyword_int_KEYWORD_NOT_FOUND();
void unit_test_read_keyword_int_KEYWORD_FOUND();

void unit_test_read_keyword_str_KEYWORD_NOT_FOUND();
void unit_test_read_keyword_str_KEYWORD_FOUND();

void unit_test_read_cell_size_CELL_SIZE_FOUND();
      
void unit_test_read_d_array_int();

void unit_test_read_d_array_real();

void unit_test_read_hierarchy_array_real();

void unit_test_read_hierarchy_array_bool();

void run_unit_tests_input();