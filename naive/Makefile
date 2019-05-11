fashion: Convolution.cu FullyConnect.cu Functions.cu Mnist.cu Net_cpu.cu \
	Net_GPU_Gemm.cu Net_GPU_Naive.cu Net_GPU_test.cu Net_GPU_Tiled.cu Pool.cu Softmax.cu
	nvcc Convolution.cu FullyConnect.cu Functions.cu Mnist.cu Net_cpu.cu \
	Net_GPU_Gemm.cu Net_GPU_Naive.cu Net_GPU_test.cu Net_GPU_Tiled.cu Pool.cu \
	Softmax.cu -o fashion -I. -w -std=c++11

clean:
	rm fashion
