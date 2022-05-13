target_sources(
	    gpu-mwdg2
		PRIVATE
		get_lvl_idx.cu
		cuda_utils.cu
		get_num_blocks.cpp
	)