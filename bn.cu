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
	int n, c, h, w;
	void set(int _n, int _c, int _h, int _w) {
		n = _n, c = _c, h = _h, w = _w;
	}
};

template <typename type>
void print(float *m, const Descript<type> &desc) {
	float buf[desc.n][desc.c][desc.h][desc.w];
	CUDA_CALL(cudaMemcpy(buf, m, sizeof(buf), cudaMemcpyDeviceToHost));
	for (int i = 0; i < desc.n; i++) {
		for (int j = 0; j < desc.c; j++) {
			for (int x = 0; x < desc.h; x++) {
				for (int y = 0; y < desc.w; y++) {
					printf("%f%c", buf[i][j][x][y], y == desc.w - 1 ? '\n' : ' ');
				}
			}
			putchar('\n');
		}
	}
}

template <typename type>
void read(float *m, const Descript<type> &desc) {
	float buf[desc.n][desc.c][desc.h][desc.w];
	for (int n = 0; n < desc.n; n++) 
		for (int c = 0; c < desc.c; c++)
			for (int i = 0; i < desc.h; i++)
				for (int j = 0; j < desc.w; j++) 
					scanf("%f", buf[n][c][i] + j);
	CUDA_CALL(cudaMemcpy(m, buf, sizeof(buf), cudaMemcpyHostToDevice));
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

	float zero = 0, one = 1, *d_in_data, *d_out_data;
	size_t sz = n * c * h * w * sizeof(float);

	CUDA_CALL(cudaMalloc(&d_in_data, sz));
	CUDA_CALL(cudaMalloc(&d_out_data, sz));

	read(d_in_data, input_output_desc);

	scanf("%d%d%d", &c, &h, &w);
	Descript<float> other_desc;
	other_desc.set(1, c, h, w);
	sz = 1 * c * h * w * sizeof(float);
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
		handle,
		// CUDNN_BATCHNORM_SPATIAL, // 1xCx1x1
		CUDNN_BATCHNORM_PER_ACTIVATION, // 1xCxHxW
		&one,
		&zero,
		cudnn_input_output_desc,
		d_in_data,
		cudnn_input_output_desc,
		d_out_data,
		cudnn_other_desc,
		d_scale,
		d_bias,
		d_mean,
		d_variance,
		CUDNN_BN_MIN_EPSILON
	));

	print(d_out_data, input_output_desc);

	CUDA_CALL(cudaFree(d_in_data));
	CUDA_CALL(cudaFree(d_out_data));
	CUDA_CALL(cudaFree(d_bias));
	CUDA_CALL(cudaFree(d_scale));
	CUDA_CALL(cudaFree(d_mean));
	CUDA_CALL(cudaFree(d_variance));

	CUDA_CALL(cudaDeviceReset());
	return 0;
}


