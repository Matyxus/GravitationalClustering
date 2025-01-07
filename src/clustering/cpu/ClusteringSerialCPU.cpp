#include "../../../headers/clustering/cpu/ClusteringSerialCPU.hpp"


// ---------------------------------------- Init ---------------------------------------- 

bool ClusteringSerialCPU::initialize(Network* network) {
	// Check if options were loaded
	std::cout << "<<<<<<< Initialitzing Network ClusteringSerialCPU >>>>>>>" << std::endl;
	if (!isInitialized()) {
		std::cout << "Unable to initialize, error during loading" << std::endl;
		return false;
	}
	// ----------------- Initialize State -----------------
	state.positions = (float2*)malloc(state.numEdges * sizeof(float2));
	state.movements = (float2*)malloc(state.numEdges * sizeof(float2));
	state.weigths = (float*)malloc(state.numEdges * sizeof(float));
	state.clusters = (int2*)malloc(state.numEdges * sizeof(int2));
	state.indexes = (int*)malloc(state.numEdges * sizeof(int));
	state.merges = (int*)malloc(state.numEdges * sizeof(int));
	state.alive = (bool *)malloc(state.numEdges * sizeof(bool));
	assert(state.isInitialized());
	// Assign values to arrays
	std::pair<float, float> position;
	for (Edge* &edge : network->getEdges()) {
		position = edge->getCentroid();
		state.alive[edge->identifier] = true;
		state.positions[edge->identifier] = { position.first, position.second };
		state.movements[edge->identifier] = { 0.f, 0.f };
		state.weigths[edge->identifier] = (edge->getCongestionIndex() + clusteringOptions->offset) * clusteringOptions->multiplier;
		state.indexes[edge->identifier] = edge->identifier;
		state.clusters[edge->identifier] = int2{ edge->identifier, -1 };
		state.merges[edge->identifier] = -1;
	}
	// ----------------- Generate grid -----------------
	if (!generateGrid(false) || !insertPoints()) {
		return false;
	}
	std::cout << "<<<<<<< Successfully initialized >>>>>>>" << std::endl;
	return true;
}


bool ClusteringSerialCPU::initialize(RandomOptions* randomOptions) {
	// Check if options were loaded
	std::cout << "<<<<<<< Initialitzing RNG ClusteringSerialCPU >>>>>>>" << std::endl;
	if (!isInitialized()) {
		std::cout << "Unable to initialize, error during loading" << std::endl;
		return false;
	}
	assert(generateRandomValues());
	// ----------------- Generate grid -----------------
	if (!generateGrid(false) || !insertPoints()) {
		return false;
	}
	std::cout << "<<<<<<< Successfully initialized >>>>>>>" << std::endl;
	return true;
}


bool ClusteringSerialCPU::generateGrid(bool re_size) {
	// TODO: check if Grid can be resized
	std::cout << "Generating GRID" << std::endl;
	const auto start = std::chrono::steady_clock::now();
	// Allocate grid arrays
	if (!grid.isInitialized()) {
		// Find borders for the grid
		const float4 borders = findBorders();
		assert(borders.x != std::numeric_limits<float>::min() && borders.y != std::numeric_limits<float>::min());
		assert(borders.z != std::numeric_limits<float>::max() && borders.w != std::numeric_limits<float>::max());
		// Assign dimensions
		grid.borders = borders;
		assert(0 <= grid.borders.z && grid.borders.z < grid.borders.x);
		assert(0 <= grid.borders.w && grid.borders.w < grid.borders.y);
		grid.rows = static_cast<int>(std::ceil((grid.borders.x - grid.borders.z) / clusteringOptions->radius));
		grid.cols = static_cast<int>(std::ceil((grid.borders.y - grid.borders.w) / clusteringOptions->radius));
		grid.size = (grid.rows * grid.cols);
		assert(grid.rows > 0 && grid.cols > 0);
		std::cout << "Grid limits are: ((" << grid.borders.z << "," << grid.borders.w << "), (" << grid.borders.x << "," << grid.borders.y << "))" << std::endl;
		std::cout << "New grid dimensions are: (" << grid.rows << "," << grid.cols << ") -> " << grid.size << std::endl;
		// Allocate array's
		grid.cells = (int*)calloc(state.numEdges, sizeof(int));
		grid.sortedCells = (int*)calloc(state.numEdges, sizeof(int));
		grid.pointOrder = (int*)calloc(state.numEdges, sizeof(int));
		grid.gridMap = (int*)calloc(grid.size, sizeof(int));
		grid.binCounts = (int*)calloc(grid.size, sizeof(int));
		// Check allocation
		assert(grid.isInitialized());
		// Assign neighbours
		grid.neighbours[2] = grid.rows;
		grid.neighbours[3] = grid.rows + 1;
		grid.neighbours[4] = grid.rows - 1;
		grid.neighbours[5] = -grid.rows;
		grid.neighbours[6] = -grid.rows + 1;
		grid.neighbours[7] = -grid.rows - 1;
	// If already allocated, zero out previous values
	} else {
		assert(grid.cells != nullptr);
		for (int i = 0; i < state.numEdges; i++) {
			grid.cells[i] = 0;
			grid.sortedCells[i] = 0;
			grid.pointOrder[i] = 0;
		}
		for (int i = 0; i < grid.size; i++) {
			grid.gridMap[i] = 0;
			grid.binCounts[i] = 0;
		}
	}
	std::cout << "Finished generating GRID, elapsed=" << since(start).count() << "[ms]" << std::endl;
	return true;
}

bool ClusteringSerialCPU::insertPoints() {
	std::cout << "Inserting " << state.numAlive << " points!" << std::endl;
	if (!grid.isInitialized()) {
		std::cout << "Error, grid is not allocated!" << std::endl;
		return false;
	}
	const auto start = std::chrono::steady_clock::now();
	// Compute cell for each edge and increase bit counts
	for (int i = 0; i < state.numEdges; i++) {
		// Skip dead point
		if (!state.alive[i]) {
			continue;
		}
		grid.cells[i] = getCell(i);
		grid.pointOrder[i] = grid.binCounts[grid.cells[i]]++;
	}
	// Compute prefixscan
	for (int i = 0; i < grid.size - 1; i++) {
		grid.gridMap[i + 1] = (grid.binCounts[i] + grid.gridMap[i]);
	}
	// Sort array by prefixscan
	for (int i = 0; i < state.numEdges; i++) {
		// Skip dead point
		if (!state.alive[i]) {
			continue;
		}
		assert(0 <= grid.gridMap[grid.cells[i]] + grid.pointOrder[i] && grid.gridMap[grid.cells[i]] + grid.pointOrder[i] < grid.size);
		grid.sortedCells[grid.gridMap[grid.cells[i]] + grid.pointOrder[i]] = i;
	}
	std::cout << "Finished inserting points, elapsed=" << since(start).count() << "[ms]" << std::endl;
	return true;
}

// ---------------------------------------- Clustering ---------------------------------------- 

bool ClusteringSerialCPU::step() {
	std::cout << "**** Peforming one step in serial mode, iteration: " << state.iteration << " ****" << std::endl;
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


bool ClusteringSerialCPU::findClusters() {
	std::cout << "Looking for closest clusters to each other" << std::endl;
	const auto start = std::chrono::steady_clock::now();
	float2 pos;
	float minDist, dist;
	int index, cellID, closestNeigh, neigh;
	// Find closest cluster to each cluster
	for (int i = 0; i < state.numEdges; i++) {
		// Skip dead point, the rest in grid must be alive, since only alive are re-inserted each iteration
		if (!state.alive[i]) {
			state.merges[i] = -1;
			continue;
		}
		minDist = std::numeric_limits<float>::max();
		pos = state.positions[i];
		cellID = grid.cells[i];
		// Go over clusters in current cell
		if (grid.binCounts[cellID] > 1) {
			for (index = grid.gridMap[cellID]; index < (grid.gridMap[cellID] + grid.binCounts[cellID]); index++) {
				neigh = grid.sortedCells[index];
				assert(state.alive[neigh]);
				// Skip the point itself
				if (neigh == i) {
					continue;
				}
				dist = getDistance2(pos, state.positions[neigh]);
				if (dist < minDist) {
					closestNeigh = neigh;
					minDist = dist;
				}
			}
		}
		// Go over clusters in neighbour cells
		for (int shift: grid.neighbours) {
			shift += cellID;
			if (!grid.isNeigh(shift)) {
				continue;
			}
			for (index = grid.gridMap[shift]; index < (grid.gridMap[shift] + grid.binCounts[shift]); index++) {
				neigh = grid.sortedCells[index];
				assert(state.alive[neigh]);
				dist = getDistance2(pos, state.positions[neigh]);
				if (dist < minDist) {
					closestNeigh = neigh;
					minDist = dist;
				}
			}
		}
		// Check if the closest one found is in merging radius (squared distance vs squared radius)
		state.merges[i] = (minDist < clusteringOptions->radius2) ? closestNeigh : -1;
	}
	std::cout << "Finished looking for closest clusters, elapsed=" << since(start).count() << "[ms]" << std::endl;
	return true;
}

bool ClusteringSerialCPU::mergeClusters() {
	std::cout << "Merging closest clusters" << std::endl;
	const auto start = std::chrono::steady_clock::now();
	const int alivePrevious = state.numAlive;
	int neigh;
	// Go over all points, we can skip checking for alive, 
	// due to "-1" being assigned to dead states in "findClusters"
	for (int i = 0; i < state.numEdges; i++) {
		neigh = state.merges[i];
		// Only lower index performs merging, to avoid duplicate merges
		if (neigh > i && i == state.merges[neigh]) {
			assert(state.alive[i] && state.alive[neigh]);
			assert(getDistance(i, neigh) < clusteringOptions->radius);
			assert(state.clusters[state.indexes[neigh]].x == state.indexes[neigh]);
			// Mark cluster as "dead"
			state.numAlive--;
			state.alive[neigh] = false;
			state.weigths[i] += state.weigths[neigh];
			// Assign cluster, requires lookup to the index table due to re-sizing
			state.clusters[state.indexes[neigh]].x = state.indexes[i];
			state.clusters[state.indexes[neigh]].y = state.iteration;
		}
	}
	std::cout << "Total merged clusters: " << (alivePrevious - state.numAlive) << std::endl;
	std::cout << "Finished merging clusters, elapsed=" << since(start).count() << "[ms]" << std::endl;
	return true;
}

bool ClusteringSerialCPU::computeMovements() {
	std::cout << "Computing cluster movements" << std::endl;
	// Check if clustering was initialized
	if (!state.isInitialized()) {
		std::cout << "Error cannot compute movements, state is not initialized!" << std::endl;
		return false;
	}
	const auto start = std::chrono::steady_clock::now();
	// Compute attraction between all points (skip the point itself -> division by zero)
	float diffX, diffY, attraction;
	float2 pos, move;
	for (int i = 0; i < state.numEdges; i++) {
		if (!state.alive[i]) {
			continue;
		}
		pos = state.positions[i];
		move = { 0.0f, 0.0f };
		// First half
		for (int j = 0; j < i; j++) {
			if (!state.alive[j]) {
				continue;
			}
			assert(!(state.positions[j].x == pos.x && state.positions[j].y == pos.y));
			diffX = (state.positions[j].x - pos.x);
			diffY = (state.positions[j].y - pos.y);
			assert(((diffX * diffX) + (diffY * diffY)) > 0);
			attraction = state.weigths[j] / ((diffX * diffX) + (diffY * diffY));
			move.x += (diffX * attraction);
			move.y += (diffY * attraction);
		}
		// Second half
		for (int j = i + 1; j < state.numEdges; j++) {
			if (!state.alive[j]) {
				continue;
			}
			assert(!(state.positions[j].x == pos.x && state.positions[j].y == pos.y));
			diffX = (state.positions[j].x - pos.x);
			diffY = (state.positions[j].y - pos.y);
			assert(((diffX * diffX) + (diffY * diffY)) > 0);
			attraction = state.weigths[j] / ((diffX * diffX) + (diffY * diffY));
			move.x += (diffX * attraction);
			move.y += (diffY * attraction);
		}
		state.movements[i] = move;
	}
	// Shift original positions by the computed movements
	for (int i = 0; i < state.numEdges; i++) {
		if (!state.alive[i]) {
			continue;
		}
		assert(state.movements[i].x != 0.0f && state.movements[i].y != 0.0f);
		state.positions[i].x += state.movements[i].x;
		state.positions[i].y += state.movements[i].y;
	}
	std::cout << "Finished computing movements, elapsed=" << since(start).count() << "[ms]" << std::endl;
	return true;
}

bool ClusteringSerialCPU::reSize() {
	// Check if we should resize
	if (!isStateResizable()) {
		std::cout << "Skipping resizing ..." << std::endl;
		return false;
	}
	assert(state.isInitialized());
	std::cout << "Resizing points to size: " << state.numAlive << " from: " << state.numEdges << std::endl;
	const auto start = std::chrono::steady_clock::now();
	// Shift all alive points in arrays to the front (0 to numAlive)
	uint32_t j = 0;
	for (int i = 0; i < state.numEdges && j < state.numAlive; i++) {
		if (state.alive[i]) {
			// We only need to change "alive", "positions", "weigths" and "indexes", 
			// others can be reset to their default values.
			state.alive[j] = true;
			state.positions[j] = state.positions[i];
			state.weigths[j] = state.weigths[i];
			state.indexes[j] = state.indexes[i];
			j++;
		}
	}
	assert(j == state.numAlive);
	// Allocate new State arrays, check for alloc failure
	float2* positions = (float2*)realloc(state.positions, state.numAlive * sizeof(float2));
	float2* movements = (float2*)realloc(state.movements, state.numAlive * sizeof(float2));
	float* weigths = (float*)realloc(state.weigths, state.numAlive * sizeof(float));
	int* merges = (int*)realloc(state.merges, state.numAlive * sizeof(int));
	int* indexes = (int*)realloc(state.indexes, state.numAlive * sizeof(int));
	bool* alive = (bool*)realloc(state.alive, state.numAlive * sizeof(bool));
	if (positions == nullptr || movements == nullptr || weigths == nullptr ||
		merges == nullptr || indexes == nullptr || alive == nullptr) {
		std::cout << "Reallocation of memory failed, possibly out of memory!" << std::endl;
		throw std::bad_alloc();
	}
	// Assign new pointers to the current ones
	state.numEdges = state.numAlive;
	state.positions = positions;
	state.movements = movements;
	state.weigths = weigths;
	state.merges = merges;
	state.indexes = indexes;
	state.alive = alive;
	std::cout << "Finished re-sizing elapsed=" << since(start).count() << "[ms]" << std::endl;
	return true;
}


float4 ClusteringSerialCPU::findBorders() {
	float4 borders = float4{
		std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
		std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
	};
	// Check
	if (!state.isInitialized()) {
		std::cout << "Error, grid or positions were not allocated!" << std::endl;
		return borders;
	}
	// Compute the smallest coordinates
	for (int i = 0; i < state.numEdges; i++) {
		// Skip dead point
		if (!state.alive[i]) {
			continue;
		}
		borders.x = std::max(borders.x, state.positions[i].x);
		borders.y = std::max(borders.y, state.positions[i].y);
		borders.z = std::min(borders.z, state.positions[i].x);
		borders.w = std::min(borders.w, state.positions[i].y);
	}
	return borders;
}
