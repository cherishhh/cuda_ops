#ifndef SOFTMAX_GPU_V4_H
#define SOFTMAX_GPU_V4_H

#include <cuda_runtime.h>

enum DeviceType {
	GPU,
	CPU
};

void initMatrix(float* data, int size);

void printMatrix(const float* vec);

void checkCudaError(const cudaError_t error_code);

void getSoftmaxTime(const float* inData, float* outData, DeviceType deviceType);

void softmaxCpu(const float* inData, float* outData);

__global__ void softmaxKernel_v1(const float* input, float* output);
__global__ void softmaxKernel_v2(const float* input, float* output);

void softmaxGpu_v1(const float* d_inData, float* d_outData);
void softmaxGpu_v2(const float* d_inData, float* d_outData);

#endif
