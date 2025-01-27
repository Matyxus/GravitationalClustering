//----------------------------------------------------------------------------------------
/**
 * \file       host.cuh
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Header for decleration of functions operating with CUDA kernels.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "kernels.cuh"
#include "alloc.cuh"

// Extern functions which can HOST use
extern "C"
{	
	//  ----- Clustering Methods ----- 
	void insertPointsCUDA(State stateGPU, Grid gridGPU, const int maxThreads, const float radiusFrac);
	void computeMovementsCUDA(State& stateGPU, const int threads, const float4& borders);
	void findClustersCUDA(State& stateGPU, Grid& gridGPU, const float radius2);
	void mergeClustersCUDA(State& stateGPU);
	float4 findBordersCUDA(State& stateGPU, const int maxThreads);
}



