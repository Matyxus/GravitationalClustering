//----------------------------------------------------------------------------------------
/**
 * \file       alloc.cuh
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Header for decleration of HOST functions operating with CUDA memory.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "utilsGPU.cuh"
#include "../base/Structs.hpp"

template <typename T> 
struct ReductionBlock {
	T** blocks = nullptr; ///< Arrays containing parallel reduction results
	T* helper = nullptr; ///< Array used as reduction output destination (when we want to perserve original)
	const int levels; ///< Total amount of reduction levels in blocks
	ReductionBlock(const int totalSize, const int threads, const bool allocateLast, const bool allocateHelper);
	~ReductionBlock();
private:
	int computeLevels(const int totalSize, const int threads, const bool allocateLast);
};

// Declare as global variable, allocated only once.
extern struct ReductionBlock<float4>* bordersBlocks;
extern struct ReductionBlock<int>* aliveBlocks;
extern struct ReductionBlock<int>* gridBlocks;

// Methods available to HOST
extern "C" 
{
	bool initializeCuda();
	// ----- Allocations ----- 
	void allocateStateCUDA(State& stateCPU, State& stateGPU);
	void allocateGridCUDA(Grid& gridGPU, const float4 borders, const int numClusters, const float radius);
	void allocateStateHelpers(const int numClusters, const int threads);
	void allocateGridHelpers(const int gridSize, const int threads);
	// ----- Freeing memory -----
	void freeStateCUDA(State& stateGPU);
	void freeGridCUDA(Grid& gridGPU);
	void freeHelpers();
	// ----- Utils ----- 
	void resetGridCUDA(Grid& gridGPU, const int numClusters);
	void copyToHost(State& stateCPU, State& stateGPU);
}


