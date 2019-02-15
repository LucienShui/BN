#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <cstdio>
#include <sys/time.h>
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

#define TIME_CALL(call)														 \
{																			 \
	double start = cpuSecond();												 \
	call;																	 \
	printf("cost %lf sec.\n", cpuSecond() - start);					 \
}

double cpuSecond() {
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
	
struct Descript {
	int n, c, h, w;
	void set(int _n, int _c, int _h, int _w) {
		n = _n, c = _c, h = _h, w = _w;
	}
};

void print(float *m, const Descript &desc) {
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

void read(float *m, const Descript &desc) {
	float buf[desc.n][desc.c][desc.h][desc.w];
	for (int n = 0; n < desc.n; n++) 
		for (int c = 0; c < desc.c; c++)
			for (int i = 0; i < desc.h; i++)
				for (int j = 0; j < desc.w; j++) 
					scanf("%f", buf[n][c][i] + j);
	CUDA_CALL(cudaMemcpy(m, buf, sizeof(buf), cudaMemcpyHostToDevice));
}

__global__ void tensor3dAdd3d(float *A, float *B, float *C, const int limit) {
	int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z * blockDim.z + threadIdx.x;
	if (z < limit) {
		int idx = x * gridDim.y * limit + y * limit + z;
		C[idx] = A[idx] + B[idx];
	}
}

__global__ void tensor1dAdd1d(float *A, float *B, float *C) {
	C[blockIdx.x] = A[blockIdx.x] + B[blockIdx.x];
}

__global__ void tensor1dAdd0d(float *A, double B, float *C) {
	C[blockIdx.x] = A[blockIdx.x] + B;
}

__global__ void tensor3dAdd0d(float *A, double B, float *C, const int limit) {
	int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z * blockDim.z + threadIdx.x;
	if (z < limit) {
		int idx = x * gridDim.y * limit + y * limit + z;
		C[idx] = A[idx] + B;
	}
}

__global__ void tensor3dSub3d(float *A, float *B, float *C, const int limit) {
	int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z * blockDim.z + threadIdx.x;
	if (z < limit) {
		int idx = x * gridDim.y * limit + y * limit + z;
		C[idx] = A[idx] - B[idx];
	}
}

__global__ void tensor3dSub1d(float *A, float *B, float *C, const int limit) {
	int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z * blockDim.z + threadIdx.x;
	if (z < limit) {
		int idx = x * gridDim.y * limit + y * limit + z;
		C[idx] = A[idx] - B[x];
	}
}

__global__ void tensor3dMul3d(float *A, float *B, float *C, const int limit) {
	int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z * blockDim.z + threadIdx.x;
	if (z < limit) {
		int idx = x * gridDim.y * limit + y * limit + z;
		C[idx] = A[idx] * B[idx];
	}
}

__global__ void tensor3dMul1d(float *A, float *B, float *C, const int limit) {
	int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z * blockDim.z + threadIdx.x;
	if (z < limit) {
		int idx = x * gridDim.y * limit + y * limit + z;
		C[idx] = A[idx] * B[x];
	}
}

__global__ void tensor1dMul1d(float *A, float *B, float *C) {
	C[blockIdx.x] = A[blockIdx.x] * B[blockIdx.x];
}

__global__ void tensor3dRsqrt(float *X, float *Y, const int limit) {
	int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z * blockDim.z + threadIdx.x;
	if (z < limit) {
		int idx = x * gridDim.y * limit + y * limit + z;
		Y[idx] = 1.0f / sqrt(X[idx]);
	}
}

__global__ void tensor1dRsqrt(float *X, float *Y) {
	Y[blockIdx.x] = 1.0f / sqrt(X[blockIdx.x]);
}

void batchNormalizationForwardInference(
		cudnnHandle_t		 handle, 
		cudnnBatchNormMode_t mode,
		const void			 *alpha,
		const void			 *beta, 
		const Descript		 &xDesc,
		const void			 *x,
		const Descript		 &yDesc,
		void				 *y,
		const Descript		 &bnScaleBiasMeanVarDesc,
		const void			 *bnScale,
		const void			 *bnBias,
		const void			 *estimatedMean,
		const void			 *estimatedVariance,
		const double		 &epsilon
	) {
	dim3 block(128);
	dim3 grid(xDesc.c, xDesc.h, (xDesc.w + block.x - 1) / block.x);

	float *u = (float *) x, *v = (float *) y, *scale, *bias;
	CUDA_CALL(cudaMalloc(&scale, 
				bnScaleBiasMeanVarDesc.c * bnScaleBiasMeanVarDesc.h * bnScaleBiasMeanVarDesc.w * sizeof(float)));
	CUDA_CALL(cudaMalloc(&bias, 
				bnScaleBiasMeanVarDesc.c * bnScaleBiasMeanVarDesc.h * bnScaleBiasMeanVarDesc.w * sizeof(float)));
	int sz = xDesc.c * xDesc.h * xDesc.w;
	if (mode == CUDNN_BATCHNORM_PER_ACTIVATION) { // 1xCxHxW
		tensor3dAdd0d <<<grid, block>>> ((float *) estimatedVariance, epsilon, scale, bnScaleBiasMeanVarDesc.w);
		tensor3dRsqrt <<<grid, block>>> (scale, scale, bnScaleBiasMeanVarDesc.w);
		tensor3dMul3d <<<grid, block>>> (scale, (float *) bnScale, scale, bnScaleBiasMeanVarDesc.w);
		tensor3dMul3d <<<grid, block>>> (scale, (float *) estimatedMean, bias, bnScaleBiasMeanVarDesc.w);
		tensor3dAdd3d <<<grid, block>>> (bias, (float *) bnBias, bias, bnScaleBiasMeanVarDesc.w);
		for (int i = 0; i < xDesc.n; i++, u += sz, v += sz) {
			tensor3dMul3d <<<grid, block>>> (u, scale, v, bnScaleBiasMeanVarDesc.w);
			tensor3dSub3d <<<grid, block>>> (v, bias, v, bnScaleBiasMeanVarDesc.w);
		}
	} else if (mode == CUDNN_BATCHNORM_SPATIAL) { // 1xCx1x1
		tensor1dAdd0d <<<bnScaleBiasMeanVarDesc.c, 1>>> ((float *) estimatedVariance, epsilon, scale);
		tensor1dRsqrt <<<bnScaleBiasMeanVarDesc.c, 1>>> (scale, scale);
		tensor1dMul1d <<<bnScaleBiasMeanVarDesc.c, 1>>> (scale, (float *) bnScale, scale);
		tensor1dMul1d <<<bnScaleBiasMeanVarDesc.c, 1>>> (scale, (float *) estimatedMean, bias);
		tensor1dAdd1d <<<bnScaleBiasMeanVarDesc.c, 1>>> (bias, (float *) bnBias, bias);
		for (int i = 0; i < xDesc.n; i++, u += sz, v += sz) {
			tensor3dMul1d <<<grid, block>>> (u, scale, v, xDesc.w);
			tensor3dSub1d <<<grid, block>>> (v, bias, v, xDesc.w);
		}
	}
}

int main() {
	int n, c, h, w;
	scanf("%d%d%d%d", &n, &c, &h, &w);
	Descript input_output_desc;
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

	float zero = 0, one = 1, *d_in_data, *d_out_data, *mine_out_data;
	size_t sz = n * c * h * w * sizeof(float);

	CUDA_CALL(cudaMalloc(&d_in_data, sz));
	CUDA_CALL(cudaMalloc(&d_out_data, sz));
	CUDA_CALL(cudaMalloc(&mine_out_data, sz));

	read(d_in_data, input_output_desc);

	scanf("%d%d%d", &c, &h, &w);
	Descript other_desc;
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

	TIME_CALL(CUDNN_CALL(cudnnBatchNormalizationForwardInference(
		handle,
		CUDNN_BATCHNORM_SPATIAL, // 1xCx1x1
		// CUDNN_BATCHNORM_PER_ACTIVATION, // 1xCxHxW
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
	)));
	CUDA_CALL(cudaDeviceSynchronize());

	TIME_CALL(batchNormalizationForwardInference(
		handle,
		CUDNN_BATCHNORM_SPATIAL, // 1xCx1x1
		// CUDNN_BATCHNORM_PER_ACTIVATION, // 1xCxHxW
		&one,
		&zero,
		input_output_desc,
		d_in_data,
		input_output_desc,
		mine_out_data,
		other_desc,
		d_scale,
		d_bias,
		d_mean,
		d_variance,
		CUDNN_BN_MIN_EPSILON
	));
	CUDA_CALL(cudaDeviceSynchronize());

	puts("cudnn:");
	print(d_out_data, input_output_desc);

	puts("mine:");
	print(mine_out_data, input_output_desc);

	CUDA_CALL(cudaFree(d_in_data));
	CUDA_CALL(cudaFree(d_out_data));
	CUDA_CALL(cudaFree(d_bias));
	CUDA_CALL(cudaFree(d_scale));
	CUDA_CALL(cudaFree(d_mean));
	CUDA_CALL(cudaFree(d_variance));

	CUDA_CALL(cudaDeviceReset());
	return 0;
}


