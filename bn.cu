#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <cstdio>
#define CUDA_CALL(call)														 \
{																			 \
	const cudaError_t error = call;											 \
	if (error != cudaSuccess) {												 \
		printf("Error: %s: %d, ", __FILE__, __LINE__);						 \
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));	 \
		std::exit(1);														 \
	}																		 \
}

#define CUDNN_CALL(call)													 \
{																			 \
	const cudnnStatus_t error = call;										 \
	if (error != CUDNN_STATUS_SUCCESS) {									 \
		printf("Error: %s: %d, ", __FILE__, __LINE__);						 \
		printf("code: %d, reason: %s\n", error, cudnnGetErrorString(error)); \
		std::exit(1);														 \
	}																		 \
}

#define NULL_CHECK(ptr)														 \
{																			 \
	if (ptr == NULL) {														 \
		printf("null ptr at line: %d\n", __LINE__);							 \
		exit(1);															 \
	}																		 \
}
	
template <typename type>
struct Descript {
	int n, c, h, w, size;
	void set(int _n, int _c, int _h, int _w) {
		n = _n, c = _c, h = _h, w = _w;
		size = n * c * h * w;
	}
};

template <typename type>
void read_(float *m, const Descript<type> &desc) {
	for (int i = 0; i < desc.n; i++) {
		for (int j = 0; j < desc.c; j++) {
			for (int x = 0; x < desc.h; x++) {
				for (int y = 0; y < desc.w; y++) scanf("%f", m + y);
				m += desc.w;
			}
		}
	}
}

template <typename type>
void read(float *m, const Descript<type> &desc) {
	int sz = desc.size;
	// int sz = desc.n * desc.c * desc.h * desc.w * sizeof(float);
	float *buf = (float *) malloc(sz);
	read_(buf, desc);
	CUDA_CALL(cudaMemcpy(m, buf, sz, cudaMemcpyHostToDevice));
	free(buf);
}

template <typename type>
void show_(float *m, const Descript<type> &desc) {
	for (int i = 0; i < desc.n; i++) {
		for (int j = 0; j < desc.c; j++) {
			for (int x = 0; x < desc.h; x++) {
				for (int y = 0; y < desc.w; y++) {
					int idx = 
						i * desc.c * desc.h * desc.w +
						j * desc.h * desc.w +
						x * desc.w + 
						y;
					printf("%f%c", m[idx], y == desc.w - 1 ? '\n' : ' ');
				}
			}
			putchar('\n');
		}
	}
}
int main() {
	int n, c, h, w;
	scanf("%d%d%d%d", &n, &c, &h, &w);
	Descript<float> input_output_desc;
	input_output_desc.set(n, c, h, w);

	cudnnHandle_t handle;
	CUDNN_CALL(cudnnCreate(&handle));

	cudnnTensorDescriptor_t cudnn_input_output_desc;
	CUDNN_CALL(cudnnCreateTensorDescriptor(&cudnn_input_output_desc));
	CUDNN_CALL(cudnnSetTensor4dDescriptor(
		cudnn_input_output_desc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		input_output_desc.n,
		input_output_desc.c,
		input_output_desc.h,
		input_output_desc.w
	));

	float zero = 0, one = 1, *d_in_data, *d_out_data, *h_out_data;
	size_t sz = n * c * h * w * sizeof(float);

	CUDA_CALL(cudaMalloc(&d_in_data, sz));
	CUDA_CALL(cudaMalloc(&d_out_data, sz));
	h_out_data = (float *) malloc(sz);

	read(d_in_data, input_output_desc);

	scanf("%d%d%d", &c, &h, &w);
	Descript<float> other_desc;
	other_desc.set(1, c, h, w);
	cudnnTensorDescriptor_t cudnn_other_desc;
	CUDNN_CALL(cudnnCreateTensorDescriptor(&cudnn_other_desc));
	CUDNN_CALL(cudnnSetTensor4dDescriptor(
		cudnn_other_desc,
		CUDNN_TENSOR_NCHW,
		CUDNN_DATA_FLOAT,
		other_desc.n,
		other_desc.c,
		other_desc.h,
		other_desc.w
	));

	float *d_bias, *d_scale, *d_mean, *d_variance;
	CUDA_CALL(cudaMalloc(&d_bias, sz));
	CUDA_CALL(cudaMalloc(&d_scale, sz));
	CUDA_CALL(cudaMalloc(&d_mean, sz));
	CUDA_CALL(cudaMalloc(&d_variance, sz));

	read(d_bias, other_desc);
	read(d_scale, other_desc);
	read(d_mean, other_desc);
	read(d_variance, other_desc);

	NULL_CHECK(&one);
	NULL_CHECK(&zero);
	NULL_CHECK(d_in_data);
	NULL_CHECK(d_out_data);
	NULL_CHECK(d_bias);
	NULL_CHECK(d_scale);
	NULL_CHECK(d_mean);
	NULL_CHECK(d_variance);

	CUDNN_CALL(cudnnBatchNormalizationForwardInference(
		handle,  // handle
		CUDNN_BATCHNORM_SPATIAL,
		// CUDNN_BATCHNORM_PER_ACTIVATION,
		&one,
		&zero,
		cudnn_input_output_desc, // xDesc
		d_in_data, 
		cudnn_input_output_desc, // yDesc
		d_out_data,
		cudnn_other_desc, // bnScaleBias
		d_bias,
		d_scale,
		d_mean,
		d_variance,
		CUDNN_BN_MIN_EPSILON
	));
	
	CUDA_CALL(cudaMemcpy(h_out_data, d_out_data, input_output_desc.size, cudaMemcpyDeviceToHost));
	show_(h_out_data, input_output_desc);

	CUDA_CALL(cudaFree(d_in_data));
	CUDA_CALL(cudaFree(d_out_data));
	CUDA_CALL(cudaFree(d_bias));
	CUDA_CALL(cudaFree(d_scale));
	CUDA_CALL(cudaFree(d_mean));
	CUDA_CALL(cudaFree(d_variance));

	CUDA_CALL(cudaDeviceReset());
	return 0;
}

/*
input data:

3 1 1 1
1 2 3

1 1 1

0
1
2
0.6667
*/
