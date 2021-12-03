#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "SaveInterval.h"

SaveInterval read_save_interval
(
	const char* input_filename, 
	const char* interval_type
);