# Clean target to remove .metal.lib and .metal.ir files
clean_ir:
	find ./mopro-msm/src/msm/metal_msm/shader -type f \( -name "*.metal.lib" -o -name "*.metal.ir" \) -delete

.PHONY: clean

# Format MSL shaders using clang-format
format_shaders:
	find mopro-msm/src/msm/metal_msm/shader -name "*.metal" -exec xcrun clang-format -i --style=WebKit {} \;
.PHONY: format_shaders
