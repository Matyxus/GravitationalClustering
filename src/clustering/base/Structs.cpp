#include "../../../headers/clustering/base/Structs.hpp"

// ------------------------------------ Grid ------------------------------------

void Grid::allocate(const float4 newBorders, const int nRows, const int nCols, const int numClusters) {
	assert(newBorders.x != std::numeric_limits<float>::min() && newBorders.y != std::numeric_limits<float>::min());
	assert(newBorders.z != std::numeric_limits<float>::max() && newBorders.w != std::numeric_limits<float>::max());
	const int newSize = nRows * nCols;
	if (!isInitialized()) { // First alloc
		// Allocate array's
		cells = (int*)malloc(numClusters * sizeof(int));
		sortedCells = (int*)malloc(numClusters * sizeof(int));
		pointOrder = (int*)malloc(numClusters * sizeof(int));
		gridMap = (int*)malloc(newSize * sizeof(int));
		binCounts = (int*)malloc(newSize * sizeof(int));
	} else { // Re-alloc
		assert(newSize < size);
		int* newCells = (int*)realloc(cells, numClusters * sizeof(int));
		int* newSortedCells = (int*)realloc(sortedCells, numClusters * sizeof(int));
		int* newPointOrder = (int*)realloc(pointOrder, numClusters * sizeof(int));
		int* newGridMap = (int*)realloc(gridMap, newSize * sizeof(int));
		int* newBinCounts = (int*)realloc(binCounts, newSize * sizeof(int));
		// Check realloc failure
		if (newCells == nullptr || newSortedCells == nullptr || newPointOrder == nullptr || 
			newGridMap == nullptr || newBinCounts == nullptr) {
			std::cout << "Reallocation of memory failed, possibly out of memory!" << std::endl;
			throw std::bad_alloc();
		}
		cells = newCells;
		sortedCells = newSortedCells;
		pointOrder = newPointOrder;
		gridMap = newGridMap;
		binCounts = newBinCounts;
	}
	// Set sizes
	size = newSize;
	rows = nRows;
	cols = nCols;
	borders = newBorders;
	// Set neighbours
	neighbours[2] = rows;
	neighbours[3] = rows + 1;
	neighbours[4] = rows - 1;
	neighbours[5] = -rows;
	neighbours[6] = -rows + 1;
	neighbours[7] = -rows - 1;
	// TODO - Check allocation -> bad alloc
	assert(isInitialized());
}


Grid::~Grid() {
	if (!host) {
		return;
	}
	std::cout << "Freeing Grid on CPU" << std::endl;
	if (cells != nullptr) {
		free(cells);
		cells = nullptr;
	}
	if (sortedCells != nullptr) {
		free(sortedCells);
		sortedCells = nullptr;
	}
	if (gridMap != nullptr) {
		free(gridMap);
		gridMap = nullptr;
	}
	if (binCounts != nullptr) {
		free(binCounts);
		binCounts = nullptr;
	}
	if (pointOrder != nullptr) {
		free(pointOrder);
		pointOrder = nullptr;
	}
}


// ------------------------------------ State ------------------------------------

void State::allocate(const int numClusters) {
	assert(numClusters > 1);
	if (!isInitialized()) { // Allocate
		positions = (float2*)malloc(numClusters * sizeof(float2));
		movements = (float2*)malloc(numClusters * sizeof(float2));
		weigths = (float*)malloc(numClusters * sizeof(float));
		clusters = (int2*)malloc(numClusters * sizeof(int2));
		indexes = (int*)malloc(numClusters * sizeof(int));
		merges = (int*)malloc(numClusters * sizeof(int));
		alive = (bool*)malloc(numClusters * sizeof(bool));
	} else { // Re-alloc
		assert(numClusters < size);
		// Allocate new State arrays, check for alloc failure
		float2* newPositions = (float2*)realloc(positions, numClusters * sizeof(float2));
		float2* newMovements = (float2*)realloc(movements, numClusters * sizeof(float2));
		float* newWeigths = (float*)realloc(weigths, numClusters * sizeof(float));
		int* newMerges = (int*)realloc(merges, numClusters * sizeof(int));
		int* newIndexes = (int*)realloc(indexes, numClusters * sizeof(int));
		bool* newAlive = (bool*)realloc(alive, numClusters * sizeof(bool));
		if (newPositions == nullptr || newMovements == nullptr || newWeigths == nullptr ||
			newMerges == nullptr || newIndexes == nullptr || newAlive == nullptr) {
			std::cout << "Reallocation of memory failed, possibly out of memory!" << std::endl;
			throw std::bad_alloc();
		}
		positions = newPositions;
		movements = newMovements;
		weigths = newWeigths;
		merges = newMerges;
		indexes = newIndexes;
		alive = newAlive;
	}
	// TODO - Check allocation -> bad alloc
	size = numClusters;
	numAlive = numClusters;
	assert(isInitialized());
}

State::~State() {
	if (!host) {
		return;
	}
	std::cout << "Freeing State on CPU" << std::endl;
	if (positions != nullptr) {
		free(positions);
		positions = nullptr;
	}
	if (movements != nullptr) {
		free(movements);
		movements = nullptr;
	}
	if (weigths != nullptr) {
		free(weigths);
		weigths = nullptr;
	}
	if (indexes != nullptr) {
		free(indexes);
		indexes = nullptr;
	}
	if (merges != nullptr) {
		free(merges);
		merges = nullptr;
	}
	if (alive != nullptr) {
		free(alive);
		alive = nullptr;
	}
	if (clusters != nullptr) {
		free(clusters);
		clusters = nullptr;
	}
}
