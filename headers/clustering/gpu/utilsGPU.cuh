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

// ------ Defines ------ 

#define CHECK_ERROR( error ) ( HandleError( error, __FILE__, __LINE__ ) )

// ------ Methods ------ 

inline bool isPowerOfTwo(int n) { return ((n & (n - 1)) == 0); }
inline int powerFloor(int x) { return 1 << std::max(0, static_cast<int>(std::ceil(std::log2(x))) - 1); }



static void HandleError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::cout << cudaGetErrorString(error) << " in " << file << " at line " << line << std::endl;
        scanf_s(" ");
        exit(EXIT_FAILURE);
    }
}

/// Structure for parallel reduction on GPU
struct Level {
	const int level; ///< Current level of reduction
	const int blocks; ///< Total number of blocks (equal to array size of that level)
	const int threads; ///< Threads per block
	const size_t sharedMem; ///< Shared memory for computation on this level
	Level(int level, int blocks, int threads, int sharedMem) : 
		level(level), blocks(blocks), threads(threads), sharedMem(sharedMem)
	{};
};

/// Structure for timing kernels on GPU 
/// (https://stackoverflow.com/questions/7876624/timing-cuda-operations)
struct GpuTimer {
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
    inline void Start() { cudaEventRecord(_start, 0); }
    inline void Stop() { cudaEventRecord(_stop, 0); }

    float Elapsed() {
        float elapsed;
        CHECK_ERROR(cudaEventSynchronize(_stop));
        CHECK_ERROR(cudaEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};

