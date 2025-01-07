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
	ClusteringParallelCPU(Config* config, Network* network) : Interface(config, network) { initialize(network); };
	ClusteringParallelCPU(Config* config, RandomOptions* randomOptions) : Interface(config, randomOptions) { initialize(randomOptions); };
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
	bool initialize(Network* network);
	bool initialize(RandomOptions* randomOptions);
	bool generateGrid(bool re_size);
	bool insertPoints();
	bool computeMovements();
	bool findClusters();
	bool mergeClusters();
	bool reSize();
	float4 findBorders();
	// Vars
	int* partial = nullptr; ///< Partial sums during parallel prefix scan
};
