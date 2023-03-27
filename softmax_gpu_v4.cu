#include <math.h>
#include <windows.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <algorithm>
#include "softmax_gpu_v4.h"

// const int nElem = 32;
const int N = 1;
// const int H = 128;
// const int W = 128;
// const int C = 64;

const int H = 3;
const int W = 3;
const int C = 64;

const int NHW = N * H * W;

void initMatrix(float* data, int size) {
    const int NMAX = 9999;
    for (int i = 0; i < size; ++i) {
        data[i] = rand() % (NMAX + 1) / float(NMAX);
    }
}

void printMatrix(const float* vec) {

    for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 3; ++w) {
            for (int c = 0; c < 5; ++c) {
                printf("%f ", vec[h * W * C + w * C + c]);
            }
        }
        printf("\n");
    }
}

void checkCudaError(const cudaError_t error_code) {
    if (error_code != cudaSuccess) {
        printf("Error: %s:%d", __FILE__, __LINE__);
        printf("error code: %d, reason: %s\n", error_code, cudaGetErrorString(error_code));
        exit(-10 * error_code);
    }
}

void getSoftmaxTime(const float* inData, float* outData, DeviceType deviceType) {
    if (deviceType == DeviceType::GPU) {
        printf("softamx gpu start. \n");
    }
    else {
        printf("softamx cpu start. \n");
    }

    float time(0);
    float duration(0);
    cudaEvent_t eventStart;
    cudaEvent_t eventEnd;
    cudaEventCreate(&eventStart);
    cudaEventCreate(&eventEnd);
    cudaEventRecord(eventStart, 0);

    int loopTime = 0;
    if (deviceType == DeviceType::GPU) {
        // loopTime = 1000;
        loopTime = 1;
    }
    else {
        loopTime = 1;
    }

    for (int i = 0; i < loopTime; ++i) {
        cudaEventRecord(eventStart, 0);

        if (deviceType == DeviceType::GPU) {
            // softmaxGpu_v1(inData, outData);
            softmaxGpu_v2(inData, outData);
        }
        else {
            softmaxCpu(inData, outData);
        }

        cudaEventRecord(eventEnd, 0);
        cudaEventSynchronize(eventEnd);
        cudaEventElapsedTime(&time, eventStart, eventEnd);
        duration += time;
    }

    float meanTime = duration / loopTime;
    if (deviceType == DeviceType::GPU) {
        printf("softmax gpu time: %.6f. \n", meanTime);
    }
    else {
        printf("softmax cpu time: %.6f. \n", meanTime);
    }

    cudaEventDestroy(eventStart);
    cudaEventDestroy(eventEnd);
    if (deviceType == DeviceType::GPU) {
        printf("softamx gpu done. \n");
    }
    else {
        printf("softamx cpu done. \n");
    }
}

void softmaxCpu(const float* inData, float* outData) {
    int shape = N * H * W * C;
    for (int i = 0; i < shape; ++i) {
        outData[i] = inData[i];
    }

    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                float expMax = outData[n * H * W * C + h * W * C + w * C];
                // get max
                for (int c = 0; c < C; ++c) {
                    expMax = max(expMax, outData[n * H * W * C + h * W * C + w * C + c]);
                }
                // sub max
                for (int c = 0; c < C; ++c) {
                    outData[n * H * W * C + h * W * C + w * C + c] -= expMax;
                }
                // exp sum
                float expSum = 0;
                for (int c = 0; c < C; ++c) {
                    expSum += exp(outData[n * H * W * C + h * W * C + w * C + c]);
                }
                // norm
                for (int c = 0; c < C; ++c) {
                    outData[n * H * W * C + h * W * C + w * C + c] = exp(outData[n * H * W * C + h * W * C + w * C + c]) / expSum;
                }
            }
        }
    }
}

__global__ void softmaxKernel_v1(const float* input, float* output) {
    int bid = blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z;
    int tid = threadIdx.x;
    int idx = bid * blockDim.x + tid;

    // get max
    __shared__ float sData[C];
    sData[tid] = input[idx];
    __syncthreads();
    for (int i = C / 2; i > 0; i /= 2) {
        if (tid < i) {
            sData[tid] = max(sData[tid], sData[tid + i]);
        }
        __syncthreads();
    }

    // sub max + exp
    output[idx] = exp(input[idx] - sData[0]);
    __syncthreads();

    // sum
    sData[tid] = output[idx];
    __syncthreads();
    for (int i = C / 2; i > 0; i /= 2) {
        if (tid < i) {
            sData[tid] += sData[tid + i];
        }
        __syncthreads();
    }

    // div
    output[idx] /= sData[0];
    __syncthreads();
}

__global__ void softmaxKernel_v2(const float* input, float* output, const int channel) {
    // dim3 grid(NHW / k); dim3 block(k * C); 1 128 128 64

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // get max
    extern __shared__ float sData[];
    sData[tid] = 0;
    if (tid < blockDim.x) {
        sData[tid] = input[idx];
        // printf("tid: %d, idx: %d, blockIdx.x: %d, blockDim.x: %d, sData[tid]: %.6f, input[idx]: %.6f \n", \
        //     tid, idx, blockIdx.x, blockDim.x, sData[tid], input[idx]);
    }
    __syncthreads();

    int k = tid / channel;
    int cid = tid % channel;
    for (int stride = channel / 2; stride > 32; stride >>= 1) {
        if (cid < stride) {
            sData[tid] = max(sData[tid + stride], sData[tid + stride]);
        }
    }
    if (cid < 32) {
        sData[tid] = max(sData[tid], sData[tid + 32]);
        sData[tid] = max(sData[tid], sData[tid + 16]);
        sData[tid] = max(sData[tid], sData[tid + 8]);
        sData[tid] = max(sData[tid], sData[tid + 4]);
        sData[tid] = max(sData[tid], sData[tid + 2]);
        sData[tid] = max(sData[tid], sData[tid + 1]);
    }
    __syncthreads();

    // sub max + exp
    output[idx] = exp(input[idx] - sData[k * channel]);
    __syncthreads();

    // sum
    for (int stride = channel / 2; stride > 32; stride >>= 1) {
        if (cid < stride) {
            sData[tid] += sData[tid + stride];
        }
    }
    if (cid < 32) {
        sData[tid] += sData[tid + 32];
        sData[tid] += sData[tid + 16];
        sData[tid] += sData[tid + 8];
        sData[tid] += sData[tid + 4];
        sData[tid] += sData[tid + 2];
        sData[tid] += sData[tid + 1];
    }

    // div
    output[idx] /= sData[k * channel];
    __syncthreads();
}

void softmaxGpu_v1(const float* d_inData, float* d_outData) {
    dim3 grid(N, H, W);
    dim3 block(C);
    softmaxKernel_v1 <<< grid, block >>> (d_inData, d_outData);
}

void softmaxGpu_v2(const float* d_inData, float* d_outData) {
    // N H W 合并, 并设定系数k，blcok不宜太小，一般为256、512
    // 128 * 128 * 64
    int k = 1;
    dim3 grid(NHW / k);
    dim3 block(k * C);
    printf("NHW: %d, C: %d \n", NHW, C);
    softmaxKernel_v2 <<< grid, block, k* C * sizeof(float) >>> (d_inData, d_outData, C);
}

int main() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    checkCudaError(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    checkCudaError(cudaSetDevice(dev));

    // initialize data at host side
    float* h_inData;
    float* h_outData;
    size_t nBytes = N * H * W * C * sizeof(float);
    h_inData = (float*)malloc(nBytes);
    h_outData = (float*)malloc(nBytes);
    int size = N * H * W * C;
    initMatrix(h_inData, size);

    printf("ori indata \n");
    printMatrix(h_inData);

    // malloc device global memory
    float* d_inData;
    cudaMalloc((float**)&d_inData, nBytes);
    float* d_outData;
    cudaMalloc((float**)&d_outData, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_inData, h_inData, nBytes, cudaMemcpyHostToDevice);
    // softmax on gpu
    getSoftmaxTime(d_inData, d_outData, DeviceType::GPU);
    // softmaxGpu(d_inData, d_outData);
    cudaMemcpy(h_outData, d_outData, nBytes, cudaMemcpyDeviceToHost);
    printf("print cuda res. \n");
    printMatrix(h_outData);
    printf("Softamx gpu test done \n");

    float* cpuOutData;
    cpuOutData = (float*)malloc(nBytes);

    getSoftmaxTime(h_inData, cpuOutData, DeviceType::CPU);
    // softmaxCpu(h_inData, cpuOutData);
    printf("print cpu res. \n");
    printMatrix(cpuOutData);
    printf("Softamx cpu test done");
}