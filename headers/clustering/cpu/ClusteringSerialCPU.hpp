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
#include <typeinfo>


/// Class that implements gravitational clustering on CPU.
/**
  This class extends Interface and implements the clustering itself,
  which is run on CPU in single threaded mode.
*/
class ClusteringSerialCPU : public Interface {
public:
	ClusteringSerialCPU(Config* config, Network* network) : Interface(config, network) { assert(isInitialized()); };
	ClusteringSerialCPU(Config* config) : Interface(config) { assert(isInitialized()); };
	~ClusteringSerialCPU(void) {};
	bool step();
private:
	bool generateGrid();
	bool insertPoints();
	bool computeMovements();
	bool findClusters();
	bool mergeClusters();
	bool countAlive();
	float4 findBorders();
};
