//----------------------------------------------------------------------------------------
/**
 * \file       kernels.cuh
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Header for decleration of CUDA kernels for clustering.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "utilsGPU.cuh"
#include "prefix.cuh"
#include "../base/Structs.hpp"

// -------------------------------- Host Kernels --------------------------------


__global__ void insertPointsKern(State stateGPU, Grid gridGPU, const float radius);
__global__ void computeMovementsKern(State stateGPU, const float radius2);


