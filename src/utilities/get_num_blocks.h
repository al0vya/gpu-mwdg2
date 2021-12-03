#pragma once

// fast ceiling of integer division
// taken from: https://stackoverflow.com/questions/2745074/fast-ceiling-of-an-integer-division-in-c-c/2745086
int get_num_blocks
(
	int num_threads, 
	int threads_per_block
);