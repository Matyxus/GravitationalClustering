//----------------------------------------------------------------------------------------
/**
 * \file       prefix.cuh
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Header for decleration of CUDA template kernels for prefix sum and parallel reduction.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "utilsGPU.cuh"

// Banks
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n)((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))


// --------------- Kernels usable by Host --------------- 

/**
	Computes borderds of grid (min/max x/y) as float4 variable from current positions of clusters.

	\param[in] input current positions of clusters
	\param[in] out Partial block output from reduction
	\param[in] alive Array flagging clusters which are still "alive"
	\param[in] size Size of positions
*/
__global__ void prescanF2(const float2* input, float4* out, const bool* alive, const int size, const int blockIndex, const int baseIndex);
/**
	Computes borderds of grid (min/max x/y) as float4 variable from lower level partial blocks.

	\param[in] input current positions of clusters
	\param[in] out Partial block output from reduction
	\param[in] size Size of positions
*/
__global__ void prescanF4(const float4* input, float4* out, const int size, const int blockIndex, const int baseIndex);
/**
	Computes prefix sum for alive clusters.

	\param[in] input current alive clusters
	\param[in] out Output array where down-sweep reduction should be stored
	\param[in] sums Partial block sums
	\param[in] size Size of input
*/
__global__ void prescanBool(const bool* input, int* out, int* sums, const int size, const int blockIndex, const int baseIndex);
/**
	Computes prefix sum for alive clusters (higher levels) or Grid.

	\param[in] input current alive clusters / grid bin counts
	\param[in] out Output array where down-sweep reduction should be stored
	\param[in] sums Partial block sums
	\param[in] size Size of input
*/
__global__ void prescanInt(const int* input, int* out, int* sums, const int size, const int blockIndex, const int baseIndex);
/**
	Simple add kernel summing block sums to original.

	\param[in] out Output array to which the sums will be added
	\param[in] sums Partial block sums
	\param[in] size Size of input
*/
__global__ void addInt(int* out, const int* sums, const int size, const int blockOffset, const int baseIndex);

