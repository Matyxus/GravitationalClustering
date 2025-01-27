#include "../../../headers/clustering/gpu/ClusteringGPU.hpp"

// ---------------------------------------- Clustering ---------------------------------------- 

bool ClusteringGPU::step() {
	std::cout << "**** Peforming one step on GPU, iteration: " << state.iteration << " ****" << std::endl;
	std::cout << "Alive: " << state.numAlive << "/" << state.size << std::endl;
	if (state.numAlive == 1) {
		std::cout << "Error: only 1 cluster is left, algorithm cannot continue" << std::endl;
		return false;
	}
	// Re-size & re-index (if possible)
	assert(countAlive());
	// Re-compute (reset) grid
	assert(generateGrid());
	// Move points (if iteration != 0)
	if (state.iteration != 0) {
		assert(computeMovements());
	}
	// Insert points to Grid
	assert(insertPoints());
	// Find clusters
	assert(findClusters());
	// Merge clusters
	assert(mergeClusters());
	state.iteration++;
	std::cout << "**** Finished ****" << std::endl;
	if (state.numAlive == 1) {
		std::cout << "Done clustering, only 1 cluster is left!" << std::endl;
	}
	return true;
}


// ---------------------------------------- Init ---------------------------------------- 

bool ClusteringGPU::initializeGPU() {
	// Check if options were loaded
	std::cout << "<<<< Initializing GPU >>>>" << std::endl;
	if (!isInitialized()) {
		std::cout << "Unable to initialize, error during loading" << std::endl;
		return false;
	} else if (!initializeCuda()) {
		return false;
	}
	allocateStateCUDA(state, stateGPU);
	allocateStateHelpers(stateGPU.size, threads);
	allocateGridCUDA(gridGPU, findBorders(), stateGPU.size, clusteringOptions->params.radius);
	allocateGridHelpers(gridGPU.size, threads);
	std::cout << "<<<< Successfully initialized GPU >>>>" << std::endl;
	return true;
}

bool ClusteringGPU::generateGrid() {
	// resetGridCUDA(gridGPU, stateGPU.size);
	return true;
}

bool ClusteringGPU::insertPoints() {
	std::cout << "Inserting " << state.numAlive << " points!" << std::endl;
	// insertPointsCUDA(stateGPU, gridGPU, threads, clusteringOptions->params.radiusFrac);
	return true;
}

bool ClusteringGPU::findClusters() {
	std::cout << "Looking for closest clusters to each other" << std::endl;
	return true;
}

bool ClusteringGPU::mergeClusters() {
	std::cout << "Merging closest clusters" << std::endl;
	return true;
}

bool ClusteringGPU::computeMovements() {
	std::cout << "Computing cluster movements" << std::endl;
	return true;
}

// ---------------------------- Utils ---------------------------- 

bool ClusteringGPU::countAlive() {
	// Check if we should resize
	if (!isStateResizable()) {
		std::cout << "Skipping resizing ..." << std::endl;
		return true;
	}
	return true;
}

