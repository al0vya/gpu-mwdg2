#pragma once

#include <cmath>

#include "../input/read_keyword_int.h"
#include "../input/read_keyword_str.h"
#include "../input/read_cell_size.h"
#include "../utilities/are_reals_equal.h"

void test_read_keyword_int_KEYWORD_NOT_FOUND();

void test_read_keyword_int_KEYWORD_FOUND();

void test_read_keyword_str_KEYWORD_NOT_FOUND();

void test_read_keyword_str_KEYWORD_FOUND();

void test_read_cell_size_CELL_SIZE_FOUND();

void run_unit_tests_input();