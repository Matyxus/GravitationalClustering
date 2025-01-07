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
#include <assert.h>
#include <typeinfo>
#include <stdexcept>


/// Class that implements gravitational clustering on CPU.
/**
  This class extends Interface and implements the clustering itself which is run on CPU,
  it provides single threaded running enviroment.
*/
class ClusteringSerialCPU : public Interface {
public:
	ClusteringSerialCPU(Config* config, Network* network) : Interface(config, network) { initialize(network); };
	ClusteringSerialCPU(Config* config, RandomOptions* randomOptions) : Interface(config, randomOptions) { initialize(randomOptions); };
	~ClusteringSerialCPU(void) {};
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
};
