# Existing Makefile content...

# Clean target to remove .metal.lib and .metal.ir files
clean_ir:
	find ./mopro-msm/src/msm/metal_msm/shader -type f \( -name "*.metal.lib" -o -name "*.metal.ir" \) -delete

.PHONY: clean
