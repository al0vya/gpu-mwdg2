
set_target_properties(gpu-mwdg2 PROPERTIES CUDA_ARCHITECTURES 61)
add_compile_definitions(_USE_TRACER=0)
add_compile_definitions(_USE_DOUBLES=0)
add_compile_definitions(_RUN_UNIT_TESTS=1)