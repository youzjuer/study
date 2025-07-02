## Set this to point to your cutlass directory, or set it as an environment
CUTLASS_DIR=/mnt/workspace/ycb/git_test/OpenMixOpl/OpenMixOpl/Z2000/third-party/cutlass

default:
	CUTLASS_DIR=${CUTLASS_DIR} pip3 install cutlass_gemm/

hopper:
	CUTLASS_DIR=${CUTLASS_DIR} COMPILE_3X_HOPPER=1 pip3 install cutlass_gemm/

clean:
	pip3 uninstall cutlass-gemm
