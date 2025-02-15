//----------------------------------------------------------------------------------------
/**
 * \file       ClusteringGPU.hpp
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Class implementing gravitational clustering algorithm on GPU (extends Interface).
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <assert.h>
#include <cmath>

// ------ Forwads ------ 
static void HandleError(cudaError_t error, const char* file, int line);

// ------ Constants ------ 

#define CHECK_ERROR( error ) ( HandleError( error, __FILE__, __LINE__ ) )
#define MAX_GPU_THREADS 256
constexpr float4 INVALID_BORDERS = float4{
    std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
    std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
};

// ------ Methods ------ 

inline bool isPowerOfTwo(int n) { return ((n & (n - 1)) == 0); }
inline int powerFloor(int x) { return 1 << std::max(0, static_cast<int>(std::ceil(std::log2(x))) - 1); }
inline int getThreadsCount(const int blocks, const int size, const int maxThreads) {
    if (blocks > 1) {
        return maxThreads;
    } else if (isPowerOfTwo(size)) {
        return size / 2;
    } // else
    return powerFloor(size);
}

static void HandleError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cout << cudaGetErrorString(error) << " in " << file << " at line " << line << std::endl;
        scanf_s(" ");
        exit(EXIT_FAILURE);
    }
}

/// Structure for recursive parallel reduction on GPU
typedef struct Level {
    dim3 grid;///< The grid of kernel
	int blocks; ///< Total number of blocks (equal to array size of that level)
	int threads; ///< Threads per block
	size_t sharedMem; ///< Shared memory for computation on this level
    // ----- Last block ----- 
    int lastBlock; ///< 0 If there is no last block (is power of 2), 1 otherwise
    int lastThreads; ///< Number of threads in last block
    size_t lastSharedMem; ///< Shared memory size of last block
    int lastElementsCount; ///< Number of elements in last block


	Level(int numElements, int maxThreads, size_t elementSize) {
        blocks = std::max(1, (int)ceil((float)numElements / (2.f * maxThreads)));
        threads = getThreadsCount(blocks, numElements, maxThreads);
        int numEltsPerBlock = threads * 2;
        sharedMem = numEltsPerBlock * elementSize;
        // Compute last block in case this is not power of 2 array size
        lastElementsCount = numElements - (blocks - 1) * numEltsPerBlock;
        lastThreads = std::max(1, lastElementsCount / 2);
        lastBlock = 0;
        lastSharedMem = 0;
        if (lastElementsCount != numEltsPerBlock) {
            lastBlock = 1;
            if (!isPowerOfTwo(lastElementsCount)) {
                lastThreads = powerFloor(lastElementsCount);
            }
            lastSharedMem = 2 * elementSize * lastThreads;
        }
        grid = dim3(std::max(1, blocks - lastBlock), 1, 1);
    };
} Level;

/// Structure for timing kernels on GPU 
/// (https://stackoverflow.com/questions/7876624/timing-cuda-operations)
typedef struct GpuTimer {
    cudaEvent_t _start; ///< Start time event
    cudaEvent_t _stop; ///< End time event
    GpuTimer() {
        CHECK_ERROR(cudaEventCreate(&_start));
        CHECK_ERROR(cudaEventCreate(&_stop));
    }
    ~GpuTimer() {
        CHECK_ERROR(cudaEventDestroy(_start));
        CHECK_ERROR(cudaEventDestroy(_stop));
    }
    inline void start() { cudaEventRecord(_start, 0); }
    inline void stop() { cudaEventRecord(_stop, 0); }

    float elapsed() {
        float elapsed;
        CHECK_ERROR(cudaEventSynchronize(_stop));
        CHECK_ERROR(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
} GpuTimer;

