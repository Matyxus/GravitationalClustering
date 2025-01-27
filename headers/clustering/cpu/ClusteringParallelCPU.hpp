//----------------------------------------------------------------------------------------
/**
 * \file       ClusteringCPU.hpp
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Class implementing gravitational clustering algorithm on CPU (extends Interface).
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "../base/Interface.hpp"
#include "../../utils.hpp"
#include <omp.h>
#include <assert.h>
#include <typeinfo>


/// Class that implements gravitational clustering on CPU.
/**
  This class extends Interface and implements the clustering itself which is run on CPU,
  it provides multi threaded running enviroment depending on given parameters.
*/
class ClusteringParallelCPU : public Interface {
public:
	ClusteringParallelCPU(Config* config, Network* network) : Interface(config, network), threads(setThreads(config->getDeviceOptions()->cpu_threads)) {
		assert(isInitialized());
		partial = (int*)calloc(threads, sizeof(int));
	};
	ClusteringParallelCPU(Config* config) : Interface(config), threads(setThreads(config->getDeviceOptions()->cpu_threads)) {
		assert(isInitialized()); 
		partial = (int*)calloc(threads, sizeof(int));
	};
	~ClusteringParallelCPU(void) {
		std::cout << "Freeing ClusteringParallelCPU" << std::endl;
		if (partial != nullptr) {
			free(partial);
			partial = nullptr;
		}
	}
	bool step();
private:
	// ------------------------ Methods ------------------------ 
	bool generateGrid();
	bool insertPoints();
	bool computeMovements();
	bool findClusters();
	bool mergeClusters();
	bool countAlive();
	float4 findBorders();
	// ------------------------ Utils ------------------------
	inline int setThreads(const int total) {
		const int maximal = omp_get_max_threads();
		if (total > maximal) {
			std::cout << "Error: Unable to use: " << total << " threads, maximum is: " << maximal << std::endl;
			return maximal;
		}
		else if (total <= 1) {
			std::cout << "Error: Allocating " << total << " threads to parallel clustering, defaulting to 1!" << std::endl;
			return 1;
		}
		std::cout << "Setting number of threads as: " << total << std::endl;
		return total;
	}
	// ------------------------ Vars ------------------------
	int* partial = nullptr; ///< Partial sums during parallel prefix scan
	const int threads; ///< Threads used by the algorithm
};
