//----------------------------------------------------------------------------------------
/**
 * \file       host.cuh
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Header for decleration of functions operating with CUDA kernels.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "../../headers/clustering/gpu/kernels.cuh"
// #include "../../headers/clustering/gpu/prefix.cuh"


// Extern functions which can HOST use
extern "C"
{	
	bool initializeCuda();
	// Allocations
	bool allocateStateCUDA(State& stateCPU, State& stateGPU);
	// Methods
	bool generateGridCUDA(Grid& gridGPU, const float4 borders, const int numEdges, const float radius);
	bool insertPointsCUDA(State& stateGPU, Grid& gridGPU, const int threads, const float radius);
;	float4 findBordersCUDA(float2* positions, bool* alive, float4** partial, struct Level* borderLevels, const int levels, int size);
}



