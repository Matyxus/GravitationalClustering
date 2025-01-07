#include "../../../headers/clustering/gpu/ClusteringGPU.hpp"



// ---------------------------------------- Init ---------------------------------------- 

bool ClusteringGPU::initialize(Network* network) {
	// Check if options were loaded
	std::cout << "<<<<<<< Initialitzing Network ClusteringGPU >>>>>>>" << std::endl;
	if (!initializeCPU(network) || !initializeGPU()) {
		return false;
	}
	std::cout << "<<<<<<< Successfully initialized >>>>>>>" << std::endl;
	return true;
}

bool ClusteringGPU::initialize(RandomOptions* randomOptions) {
	// Check if options were loaded
	std::cout << "<<<<<<< Initialitzing RNG ClusteringGPU >>>>>>>" << std::endl;
	if (!initializeCPU(randomOptions) || !initializeGPU()) {
		return false;
	}
	std::cout << "<<<<<<< Successfully initialized >>>>>>>" << std::endl;
	return true;
}


bool ClusteringGPU::initializeCPU(Network* network) {
	// Check if options were loaded
	std::cout << "Initializing CPU" << std::endl;
	if (!isInitialized()) {
		std::cout << "Unable to initialize, error during loading" << std::endl;
		return false;
	}
	// ----------------- Initialize State -----------------
	assert(state.numEdges == network->getEdges().size());
	state.positions = (float2*)malloc(state.numEdges * sizeof(float2));
	state.movements = (float2*)malloc(state.numEdges * sizeof(float2));
	state.weigths = (float*)malloc(state.numEdges * sizeof(float));
	state.clusters = (int2*)malloc(state.numEdges * sizeof(int2));
	state.indexes = (int*)malloc(state.numEdges * sizeof(int));
	state.merges = (int*)malloc(state.numEdges * sizeof(int));
	state.alive = (bool*)malloc(state.numEdges * sizeof(bool));
	assert(state.isInitialized());
	// Assign values to arrays
	std::pair<float, float> position;
	for (Edge*& edge : network->getEdges()) {
		position = edge->getCentroid();
		state.alive[edge->identifier] = true;
		state.positions[edge->identifier] = { position.first, position.second };
		state.movements[edge->identifier] = { 0.f, 0.f };
		state.weigths[edge->identifier] = edge->getCongestionIndex() * clusteringOptions->multiplier;
		state.indexes[edge->identifier] = edge->identifier;
		state.clusters[edge->identifier] = int2{ edge->identifier, -1 };
		state.merges[edge->identifier] = -1;
	}
	std::cout << "Successfully initialized CPU" << std::endl;
	return true;
}

bool ClusteringGPU::initializeCPU(RandomOptions* randomOptions) {
	// Check if options were loaded
	std::cout << "Initialitzing RNG CPU" << std::endl;
	if (!isInitialized()) {
		std::cout << "Unable to initialize, error during loading" << std::endl;
		return false;
	}
	assert(generateRandomValues());
	std::cout << "Successfully initialized CPU" << std::endl;
	return true;
}

bool ClusteringGPU::initializeGPU() {
	// Check if options were loaded
	std::cout << "Initializing GPU" << std::endl;
	if (!isInitialized()) {
		std::cout << "Unable to initialize, error during loading" << std::endl;
		return false;
	} else if (!initializeCuda()) {
		return false;
	} else if (!allocateStateCUDA(state, stateGPU)) {
		return false;
	} else  if (!generateGrid(false) || !insertPoints()) {
		return false;
	}
	std::cout << "Successfully initialized GPU" << std::endl;
	return true;
}


bool ClusteringGPU::generateGrid(bool re_size) {
	// TODO: check if Grid can be resized
	std::cout << "Generating GRID" << std::endl;
	// preallocBlockSums(state.numEdges);
	initializeBlocksBorderds();
	const float4 borderds = findBorders();
	return generateGridCUDA(gridGPU, borderds, state.numEdges, clusteringOptions->radius);
}

bool ClusteringGPU::insertPoints() {
	std::cout << "Inserting " << state.numAlive << " points!" << std::endl;

	return insertPointsCUDA(stateGPU, gridGPU, clusteringOptions->device.threads, clusteringOptions->radius);
}

// ---------------------------------------- Clustering ---------------------------------------- 

bool ClusteringGPU::step() {
	std::cout << "**** Peforming one step on GPU, iteration: " << state.iteration << " ****" << std::endl;
	std::cout << "Alive: " << state.numAlive << "/" << state.numEdges << std::endl;
	if (state.numAlive == 1) {
		std::cout << "Error: only 1 cluster is left, algorithm cannot continue" << std::endl;
		return false;
	}
	// Find clusters
	assert(findClusters());
	// Merge clusters
	assert(mergeClusters());
	if (state.numAlive == 1) {
		std::cout << "Finished algorithm, only 1 cluster is left!" << std::endl;
		return true;
	}
	// Re-size & re-index (if needed)
	const bool re_size = reSize();
	// Compute shifts
	assert(computeMovements());
	// Re-compute (reset) grid and insert points again
	assert(generateGrid(re_size));
	assert(insertPoints());
	state.iteration++;
	std::cout << "**** Finished ****" << std::endl;
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

bool ClusteringGPU::reSize() {
	// Check if we should resize
	if (!isStateResizable()) {
		std::cout << "Skipping resizing ..." << std::endl;
		return false;
	}

	return true;
}


float4 ClusteringGPU::findBorders() {
	return findBordersCUDA(stateGPU.positions, stateGPU.alive, partialBorders, levelsBorders.data(), borderLevels, state.numEdges);
}

// -------------------------- Blocks -------------------------- 

bool ClusteringGPU::initializeBlocksBorderds() {
	// Compute the new parameters based on the current number of clusters
	const float work = 2.0f * clusteringOptions->device.threads; // Total amount of work per threads in block
	int size = state.numEdges;
	if (size <= 1) {
		std::cout << "Error, cannot allocate bocks for borderds for size: " << size << std::endl;
		return false;
	}
	// Realloc in case we are allocating again
	if (partialBorders != nullptr) {
		std::cout << "Freeing previous blocks for borders" << std::endl;
		freeBlocksBorders();
	}

	borderLevels = 0;
	// Compute the total number of levels needed
	while (size > 1) {
		int numBlocks = std::max(1, static_cast<int>(ceil((float) size / work)));
		std::cout << "Level: " << borderLevels << " has: " << numBlocks << " blocks" << std::endl;
		borderLevels++;
		size = numBlocks;
	}
	// New alloc
	if (partialBorders == nullptr) {
		std::cout << "First time allocating partialBorderds" << std::endl;
		assert(levelsBorders.size() == 0);
		partialBorders = (float4**) malloc(borderLevels * sizeof(float4*));
		assert(partialBorders != nullptr);
	}
	levelsBorders.clear();
	// Allocate GPU and structures
	size = state.numEdges;
	borderLevels = 0;
	int numThreads;
	while (size > 1) {
		int numBlocks = std::max(1, static_cast<int>(ceil((float)size / work)));
		if (numBlocks > 1) {
			numThreads = clusteringOptions->device.threads;
		} else { // Only one block, lower the number of threads
			numThreads = powerFloor(size);
			assert(numThreads < size && numThreads >= 1);
			if (isPowerOfTwo(size) && size != 1) {
				assert(numThreads * 2 == size);
			}
		}
		std::cout << "Level: " << borderLevels << " has: " << numBlocks << " blocks" << " threads: " << numThreads << std::endl;
		CHECK_ERROR(cudaMalloc((void**)&partialBorders[borderLevels], numBlocks * sizeof(float4)));
		levelsBorders.push_back(Level(borderLevels, numBlocks, numThreads, 2 * numThreads * sizeof(float4)));
		borderLevels++;
		size = numBlocks;
	}
	return true;
}

bool ClusteringGPU::initializeBlocksAlive() {
	// Realloc in case we are allocating again
	if (partialBorders != nullptr) {
		freeBlocksAlive();
	}
	// Compute the new parameters based on the current number of clusters

	return true;
}

bool ClusteringGPU::initializeBlocksGrid() {
	// Realloc in case we are allocating again
	if (partialBorders != nullptr) {
		freeBlocksGrid();
	}
	// Compute the new parameters based on the current Grid size

	return true;
}



// -------------------------- Free Memory -------------------------- 

void ClusteringGPU::freeBlocksBorders() {
	if (partialBorders != nullptr) {
		assert(levelsBorders.size() > 0 && borderLevels > 0);
		for (int i = 0; i < borderLevels; i++) {
			CHECK_ERROR(cudaFree(partialBorders[i]));
		}
	}
}

void ClusteringGPU::freeBlocksAlive() {
	if (partialAlive != nullptr) {
		assert(levelsAlive.size() > 0 && aliveLevels > 0);
		for (int i = 0; i < aliveLevels; i++) {
			CHECK_ERROR(cudaFree(partialAlive[i]));
		}
	}
}


void ClusteringGPU::freeBlocksGrid() {
	if (partialGrid != nullptr) {
		assert(levelsGrid.size() > 0 && gridLevels > 0);
		for (int i = 0; i < gridLevels; i++) {
			CHECK_ERROR(cudaFree(partialGrid[i]));
		}
	}
}

// -------------------------- Destructor -------------------------- 

ClusteringGPU::~ClusteringGPU() {
	// ------------------ Grid ------------------
	std::cout << "Freeing GPU Grid" << std::endl;
	if (gridGPU.cells != nullptr) {
		CHECK_ERROR(cudaFree(gridGPU.cells));
		gridGPU.cells = nullptr;
	}
	if (gridGPU.sortedCells != nullptr) {
		CHECK_ERROR(cudaFree(gridGPU.sortedCells));
		gridGPU.sortedCells = nullptr;
	}
	if (gridGPU.gridMap != nullptr) {
		CHECK_ERROR(cudaFree(gridGPU.gridMap));
		gridGPU.gridMap = nullptr;
	}
	if (gridGPU.binCounts != nullptr) {
		CHECK_ERROR(cudaFree(gridGPU.binCounts));
		gridGPU.binCounts = nullptr;
	}
	if (gridGPU.pointOrder != nullptr) {
		CHECK_ERROR(cudaFree(gridGPU.pointOrder));
		gridGPU.pointOrder = nullptr;
	}
	// ------------------ Reductions ------------------
	std::cout << "Freeing GPU helper vars" << std::endl;
	if (partialBorders != nullptr) {
		freeBlocksBorders();
		// Free CPU
		free(partialBorders);
		partialBorders = nullptr;
	}
	if (partialAlive != nullptr) {
		freeBlocksAlive();
		// Free CPU
		free(partialAlive);
		partialBorders = nullptr;
	}
	if (partialGrid != nullptr) {
		freeBlocksGrid();
		// Free CPU
		free(partialGrid);
		partialGrid = nullptr;
	}
	// ------------------ State ------------------
	std::cout << "Freeing GPU State" << std::endl;
	if (stateGPU.positions != nullptr) {
		CHECK_ERROR(cudaFree(stateGPU.positions));
		stateGPU.positions = nullptr;
	}
	if (stateGPU.movements != nullptr) {
		CHECK_ERROR(cudaFree(stateGPU.movements));
		stateGPU.movements = nullptr;
	}
	if (stateGPU.weigths != nullptr) {
		CHECK_ERROR(cudaFree(stateGPU.weigths));
		stateGPU.weigths = nullptr;
	}
	if (stateGPU.indexes != nullptr) {
		CHECK_ERROR(cudaFree(stateGPU.indexes));
		stateGPU.indexes = nullptr;
	}
	if (stateGPU.merges != nullptr) {
		CHECK_ERROR(cudaFree(stateGPU.merges));
		stateGPU.merges = nullptr;
	}
	if (stateGPU.alive != nullptr) {
		CHECK_ERROR(cudaFree(stateGPU.alive));
		stateGPU.alive = nullptr;
	}
	if (stateGPU.clusters != nullptr) {
		CHECK_ERROR(cudaFree(stateGPU.clusters));
		stateGPU.clusters = nullptr;
	}
}







