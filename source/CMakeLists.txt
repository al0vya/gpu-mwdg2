add_subdirectory(input)
add_subdirectory(mra)
add_subdirectory(neighbours)
add_subdirectory(operators)
add_subdirectory(output)
add_subdirectory(utilities)
add_subdirectory(zorder)

target_sources(
	    gpu-mwdg2
		PRIVATE
		main.cu
	)