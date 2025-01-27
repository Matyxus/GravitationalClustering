#include "../../../headers/clustering/cpu/ClusteringSerialCPU.hpp"

// ---------------------------------------- Step ---------------------------------------- 

bool ClusteringSerialCPU::step() {
	std::cout << "**** Peforming one step in serial mode, iteration: " << state.iteration << " ****" << std::endl;
	std::cout << "Alive: " << state.numAlive << "/" << state.size << std::endl;
	if (state.numAlive == 1) {
		std::cout << "Error: only 1 cluster is left, algorithm cannot continue" << std::endl;
		return false;
	}
	// Create new timer
	record = PeformanceRecord();
	// ------ Re-size & re-index (if possible) ------ 
	auto start = std::chrono::steady_clock::now();
	assert(countAlive());
	auto result = since(start);
	record.stateResizeElapsed = result.count();
	std::cout << "1) Finished re-sizing State, elapsed=" << toMS(result) << "[ms]" << std::endl;
	// ------ Re-compute (reset) grid ------ 
	start = std::chrono::steady_clock::now();
	assert(generateGrid());
	result = since(start);
	record.gridResizeElapsed = result.count();
	std::cout << "2) Finished re-sizing Grid, elapsed=" << toMS(result) << "[ms]" << std::endl;
	// ------ Move points (if iteration != 0) ------ 
	if (state.iteration != 0) {
		start = std::chrono::steady_clock::now();
		assert(computeMovements());
		result = since(start);
		record.movementsElapsed = result.count();
		std::cout << "3) Finished computing movements, elapsed=" << toMS(result) << "[ms]" << std::endl;
	}
	// ------ Insert points to Grid------ 
	start = std::chrono::steady_clock::now();
	assert(insertPoints());
	result = since(start);
	record.insertElapsed = result.count();
	std::cout << "4) Finished inserting " << state.numAlive << " clusters, elapsed=" << toMS(result) << "[ms]" << std::endl;
	// ------  Find clusters ------ 
	start = std::chrono::steady_clock::now();
	assert(findClusters());
	result = since(start);
	record.neighboursElapsed = result.count();
	std::cout << "5) Finished finding clusters, elapsed=" << toMS(result) << "[ms]" << std::endl;
	// ------ Merge clusters ------ 
	start = std::chrono::steady_clock::now();
	assert(mergeClusters());
	result = since(start);
	record.mergingElapsed = result.count();
	std::cout << "6) Finished merging clusters, elapsed=" << toMS(result) << "[ms]" << std::endl;
	state.iteration++;
	std::cout << "**** Finished ****" << std::endl;
	if (state.numAlive == 1) {
		std::cout << "Done clustering, only 1 cluster is left!" << std::endl;
	}
	records.push_back(record);
	return true;
}

// ---------------------------------------- Methods ---------------------------------------- 

bool ClusteringSerialCPU::generateGrid() {
	// Check for resizing, always true on 1st iteration
	if (((state.iteration % clusteringOptions->gridResize.frequency) == 0)) {
		// Find borders for the grid
		const float4 borders = findBorders();
		// Compute dimensions
		const int rows = static_cast<int>(std::ceil((borders.x - borders.z) * clusteringOptions->params.radiusFrac));
		const int cols = static_cast<int>(std::ceil((borders.y - borders.w) * clusteringOptions->params.radiusFrac));
		assert(rows > 0 && cols > 0);
		// Decide if we are allocating or re-allocating
		if (!grid.isInitialized() || isGridResizable(rows * cols)) {
			grid.allocate(borders, rows, cols, state.size);
			std::cout << "Grid limits are: ((" << grid.borders.z << "," << grid.borders.w << "), (" << grid.borders.x << "," << grid.borders.y << "))" << std::endl;
			std::cout << "New grid dimensions are: (" << grid.rows << "," << grid.cols << ") -> " << grid.size << std::endl;
		}
	}
	assert(grid.isInitialized());
	// Reset values
	for (int i = 0; i < state.size; i++) {
		grid.cells[i] = 0;
		grid.sortedCells[i] = 0;
		grid.pointOrder[i] = 0;
	}
	for (int i = 0; i < grid.size; i++) {
		grid.gridMap[i] = 0;
		grid.binCounts[i] = 0;
	}
	return true;
}

bool ClusteringSerialCPU::insertPoints() {
	if (!grid.isInitialized()) {
		std::cout << "Error, grid is not allocated!" << std::endl;
		return false;
	}
	// Compute cell for each edge and increase bit counts
	for (int i = 0; i < state.size; i++) {
		if (!state.alive[i]) { // Skip dead cluster
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
	for (int i = 0; i < state.size; i++) {
		if (!state.alive[i]) { // Skip dead cluster
			continue;
		}
		assert(0 <= grid.gridMap[grid.cells[i]] + grid.pointOrder[i] && grid.gridMap[grid.cells[i]] + grid.pointOrder[i] < grid.size);
		grid.sortedCells[grid.gridMap[grid.cells[i]] + grid.pointOrder[i]] = i;
	}
	return true;
}

bool ClusteringSerialCPU::findClusters() {
	float2 pos;
	float minDist, dist;
	int index, cellID, closestNeigh, neigh;
	// Find closest cluster to each cluster
	for (int i = 0; i < state.size; i++) {
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
		state.merges[i] = (minDist < clusteringOptions->params.radius2) ? closestNeigh : -1;
	}
	return true;
}

bool ClusteringSerialCPU::mergeClusters() {
	const int alivePrevious = state.numAlive;
	int neigh;
	// Go over all points, we can skip checking for alive, 
	// due to "-1" being assigned to dead states in "findClusters"
	for (int i = 0; i < state.size; i++) {
		neigh = state.merges[i];
		// Only lower index performs merging, to avoid duplicate merges
		if (neigh > i && i == state.merges[neigh]) {
			assert(state.alive[i] && state.alive[neigh]);
			assert(getDistance(i, neigh) < clusteringOptions->params.radius);
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
	return true;
}

bool ClusteringSerialCPU::computeMovements() {
	// Check if clustering was initialized
	if (!state.isInitialized()) {
		std::cout << "Error cannot compute movements, state is not initialized!" << std::endl;
		return false;
	}
	// Compute attraction between all points (skip the point itself -> division by zero)
	float diffX, diffY, attraction;
	float2 pos, move;
	for (int i = 0; i < state.size; i++) {
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
			diffX = (state.positions[j].x - pos.x);
			diffY = (state.positions[j].y - pos.y);
			assert(((diffX * diffX) + (diffY * diffY)) > 0);
			attraction = state.weigths[j] / ((diffX * diffX) + (diffY * diffY));
			move.x += (diffX * attraction);
			move.y += (diffY * attraction);
		}
		// Second half
		for (int j = i + 1; j < state.size; j++) {
			if (!state.alive[j]) {
				continue;
			}
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
	for (int i = 0; i < state.size; i++) {
		if (!state.alive[i]) {
			continue;
		}
		// Clips value between borders in case of high attraction
		state.positions[i].x = clip(state.positions[i].x + state.movements[i].x, grid.borders.z, grid.borders.x);
		state.positions[i].y = clip(state.positions[i].y + state.movements[i].y, grid.borders.w, grid.borders.y);
	}
	return true;
}

// ---------------------------------------- Utils ---------------------------------------- 

bool ClusteringSerialCPU::countAlive() {
	// Check if we should resize
	assert(state.isInitialized());
	if (!isStateResizable()) {
		std::cout << "Skipping resizing ..." << std::endl;
		return true;
	}
	std::cout << "Resizing clusters to size: " << state.numAlive << " from: " << state.size << std::endl;
	// Shift all alive clusters in arrays to the front (0 to numAlive)
	int j = 0;
	for (int i = 0; i < state.size && j < state.numAlive; i++) {
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
	state.allocate(state.numAlive);
	return true;
}


float4 ClusteringSerialCPU::findBorders() {
	std::cout << "Looking for Grid borders" << std::endl;
	const auto start = std::chrono::steady_clock::now();
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
	for (int i = 0; i < state.size; i++) {
		// Skip dead point
		if (!state.alive[i]) {
			continue;
		}
		borders.x = std::max(borders.x, state.positions[i].x);
		borders.y = std::max(borders.y, state.positions[i].y);
		borders.z = std::min(borders.z, state.positions[i].x);
		borders.w = std::min(borders.w, state.positions[i].y);
	}
	assert(borders.x != std::numeric_limits<float>::min() && borders.y != std::numeric_limits<float>::min());
	assert(borders.z != std::numeric_limits<float>::max() && borders.w != std::numeric_limits<float>::max());
	assert(0.f <= borders.z && borders.z < borders.x);
	assert(0.f <= borders.w && borders.w < borders.y);
	const auto result = since(start);
	record.bordersElapsed = result.count();
	std::cout << "Finished finding borders, elapsed=" << toMS(result) << "[ms]" << std::endl;
	return borders;
}
