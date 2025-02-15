//----------------------------------------------------------------------------------------
/**
 * \file       ClusteringGPU.hpp
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Class implementing gravitational clustering algorithm on GPU (extends Interface).
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "../base/Interface.hpp"
#include "host.cuh"
#include <assert.h>
#include <typeinfo>
#include <stdexcept>

/// Class that implements gravitational clustering on GPU.
/**
  This class extends Interface and implements the clustering itself which is run on GPU,
  provides method to copy data back to CPU.
*/
class ClusteringGPU : public Interface {
public:
	ClusteringGPU(Config* config, Network* network) : Interface(config, network), threads(setThreads(config->getDeviceOptions()->gpu_threads)) { assert(initializeGPU()); };
	ClusteringGPU(Config* config) : Interface(config), threads(setThreads(config->getDeviceOptions()->gpu_threads)) { assert(initializeGPU()); };
	~ClusteringGPU() {
		freeStateCUDA(stateGPU);
		freeGridCUDA(gridGPU);
		freeHelpers();
	}
	bool step();
	void copyToCPU() { copyToHost(state, stateGPU); };
private:
	// ------------------------ Methods ------------------------ 
	// Clustering
	bool generateGrid();
	bool insertPoints();
	bool computeMovements();
	bool findClusters();
	bool mergeClusters();
	bool countAlive();
	float4 findBorders() { return findBordersCUDA(stateGPU, threads);};
	// Utils
	bool initializeGPU();
	inline int setThreads(const int total) {
		if (!isPowerOfTwo(total)) {
			std::cout << "Erorr: Number of GPU threads must be 2^X, got: " << total;
			std::cout << ", defaulting to: " << MAX_GPU_THREADS << std::endl;
			return MAX_GPU_THREADS;
		}
		std::cout << "Setting number of GPU threads as: " << total << std::endl;
		return total;
	}
	// ------------------------ GPU Vars ------------------------ 
	const int threads; ///< Maximal amount of threads per block on GPU
	State stateGPU = State(false); ///< State of current gravitational clustering on GPU
	Grid gridGPU = Grid(false); ///< Grid spatial structure on GPU
};
