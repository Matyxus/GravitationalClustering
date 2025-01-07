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
  always after every iteration data from GPU is transferred to CPU for checking/rendering.
*/
class ClusteringGPU : public Interface {
public:
	ClusteringGPU(Config* config, Network* network) : Interface(config, network) { initialize(network); };
	ClusteringGPU(Config* config, RandomOptions* randomOptions) : Interface(config, randomOptions) { initialize(randomOptions); };
	~ClusteringGPU();
	bool step();
private:
	// ------------------------ Methods ------------------------ 
	// Init's
	bool initialize(Network* network);
	bool initialize(RandomOptions* randomOptions);
	bool initializeCPU(Network* network);
	bool initializeCPU(RandomOptions* randomOptions);
	bool initializeGPU();
	// Clustering
	bool generateGrid(bool re_size);
	bool insertPoints();
	bool computeMovements();
	bool findClusters();
	bool mergeClusters();
	bool reSize();
	float4 findBorders();
	// Utils
	bool initializeBlocksBorderds();
	bool initializeBlocksAlive();
	bool initializeBlocksGrid();
	/**
	  Frees parallel reduction blocks for finding borderds on GPU.

	  Does not however free CPU allocation of arrays.
	*/
	void freeBlocksBorders();
	/**
	  Frees parallel reduction blocks for finding borderds on GPU.

	  Does not however free CPU allocation of arrays.
	*/
	void freeBlocksAlive();
	/**
	  Frees parallel reduction blocks for Grid sorting on GPU.

	  Does not however free CPU allocation of arrays.
	*/
	void freeBlocksGrid();
	// ------------------------ GPU Vars ------------------------ 
	State stateGPU = State(false); ///< State of current gravitational clustering on GPU
	Grid gridGPU = Grid(false); ///< Grid spatial structure on GPU
	// Borders reduction
	float4** partialBorders = nullptr; ///< Partial arrays containing parallel reduction results for findind borderds
	std::vector<Level> levelsBorders; ///< Supporting structures containing vars of parallel reduction for findind borderds
	int borderLevels = 0; ///< Total number of levels for reduction on borders
	// Alive reduction
	int** partialAlive = nullptr; ///< Partial arrays containing parallel reduction results for counting alive clusters
	std::vector<Level> levelsAlive; ///< Supporting structures containing vars of parallel reduction for counting alive
	int aliveLevels = 0; ///< Total number of levels for reduction on counting alive clusters
	// Grid reduction
	int** partialGrid = nullptr; ///< Partial arrays containing parallel reduction for sorting Grid by cells
	std::vector<Level> levelsGrid; ///< Supporting structures containing vars of parallel reduction on Grid
	int gridLevels = 0; ///< Total number of levels for reduction on sorting grid

};
