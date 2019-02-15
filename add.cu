#include <cuda.h>
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

__global__ void matrix3dAdd(float *A, float *B, float *C, const int limit) {
	int x = blockIdx.x, y = blockIdx.y, z = blockIdx.z * blockDim.z + threadIdx.x;
	if (z < limit) {
		int idx = x * gridDim.y * limit + y * limit + z;
		C[idx] = A[idx] + B[idx];
	}
}

struct Shape {
	int c, h, w;
} shape;

void addOnGPU(float *A, float *B, float *&C, const Shape &shape) {
	CUDA_CALL(cudaMalloc(&C, shape.c * shape.h * shape.w * sizeof(float)));
	dim3 block(128);
	dim3 grid(shape.c, shape.h, (shape.w + block.x - 1) / block.x);
	matrix3dAdd <<<block, grid>>> (A, B, C, shape.w);
}

void read(float *&m, const Shape &shape) {
	float buf[shape.c][shape.h][shape.w];
	for (int i = 0; i < shape.c; i++) 
		for (int j = 0; j < shape.h; j++)
			for (int k = 0; k < shape.w; k++)
				scanf("%f", buf[i][j] + k);
	CUDA_CALL(cudaMalloc(&m, sizeof(buf)));
	CUDA_CALL(cudaMemcpy(m, buf, sizeof(buf), cudaMemcpyHostToDevice));
}

void print(float *m, const Shape &shape) {
	float buf[shape.c][shape.h][shape.w];
	CUDA_CALL(cudaMemcpy(buf, m, sizeof(buf), cudaMemcpyDeviceToHost));
	for (int i = 0; i < shape.c; i++) 
		for (int j = 0; j < shape.h; j++)
			for (int k = 0; k < shape.w; k++)
				printf("%f%c", buf[i][j][k], k == shape.w - 1 ? '\n' : ' ');
}

int main() {
	scanf("%d%d%d", &shape.c, &shape.h, &shape.w);
	float *A, *B, *C;
	read(A, shape);
	read(B, shape);
	addOnGPU(A, B, C, shape);
	print(C, shape);
	return 0;
}
