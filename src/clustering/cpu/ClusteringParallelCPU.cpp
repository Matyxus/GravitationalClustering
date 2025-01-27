#include "../../../headers/clustering/cpu/ClusteringParallelCPU.hpp"

// ---------------------------------------- Step ---------------------------------------- 

bool ClusteringParallelCPU::step() {
	std::cout << "**** Peforming one step in parallel mode, iteration: " << state.iteration << " ****" << std::endl;
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

bool ClusteringParallelCPU::generateGrid() {
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
	// Reset values
	assert(grid.isInitialized());
	omp_set_dynamic(0);
	#pragma omp parallel for num_threads(threads)
	for (int i = 0; i < state.size; i++) {
		grid.cells[i] = 0;
		grid.sortedCells[i] = 0;
		grid.pointOrder[i] = 0;
	}
	omp_set_dynamic(0);
	#pragma omp parallel for num_threads(threads)
	for (int i = 0; i < grid.size; i++) {
		grid.gridMap[i] = 0;
		grid.binCounts[i] = 0;
	}
	return true;
}

bool ClusteringParallelCPU::insertPoints() {
	if (!grid.isInitialized()) {
		std::cout << "Error, grid is not allocated!" << std::endl;
		return false;
	}
	// Compute cell for each edge and increase bit counts
	omp_set_dynamic(0);
	#pragma omp parallel for num_threads(threads)
	for (int i = 0; i < state.size; i++) {
		// Skip dead point
		if (!state.alive[i]) {
			continue;
		}
		grid.cells[i] = getCell(i);
		#pragma omp atomic capture
		grid.pointOrder[i] = grid.binCounts[grid.cells[i]]++;
	}
	// Compute prefixscan, just reduction or sum into partials, then sum partials to the rest, barrier approach or chunk with reduction...
	const int work = grid.size / threads;
	assert(work > 0);
	omp_set_dynamic(0);
	#pragma omp parallel num_threads(threads)
	{
		const int tid = omp_get_thread_num();
		const int a = tid * work;
		const int b = (tid == threads - 1) ? grid.size : (a + work);
		partial[tid] = 0; // Reset partial sums
		for (int i = a; i < b - 1; i++) {
			grid.gridMap[i + 1] = (grid.binCounts[i] + grid.gridMap[i]);
		}
		// Write partial sums and sync all threads
		partial[tid] = grid.gridMap[b - 1];
		#pragma omp barrier
		// Add partial sums to indexes
		int offset = 0;
		for (int i = 0; i < tid; i++) {
			offset += partial[i];
		}
		// Skip empty partials
		if (offset != 0) {
			for (int i = a; i < b; i++) {
				grid.gridMap[i] += offset;
			}
		}
	}
	// Sort array by prefixscan
	omp_set_dynamic(0);
	#pragma omp parallel for num_threads(threads)
	for (int i = 0; i < state.size; i++) {
		// Skip dead point
		if (!state.alive[i]) {
			continue;
		}
		assert(0 <= grid.gridMap[grid.cells[i]] + grid.pointOrder[i] && grid.gridMap[grid.cells[i]] + grid.pointOrder[i] < grid.size);
		grid.sortedCells[grid.gridMap[grid.cells[i]] + grid.pointOrder[i]] = i;
	}
	return true;
}

bool ClusteringParallelCPU::findClusters() {
	float2 pos;
	float minDist, dist;
	int index, cellID, closestNeigh, neigh, shift;
	// Find closest cluster to each cluster
	omp_set_dynamic(0);
	#pragma omp parallel for private(pos, minDist, dist, index, cellID, closestNeigh, neigh, shift) num_threads(threads)
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
		for (int j = 0; j < 8; j++) {
			shift = cellID + grid.neighbours[j];
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

bool ClusteringParallelCPU::mergeClusters() {
	int numAlive = state.numAlive;
	int neigh;
	// Go over all points, we can skip checking for alive, 
	// due to "-1" being assigned to dead states in "findClusters"
	omp_set_dynamic(0);
	#pragma omp parallel for private(neigh) reduction(-:numAlive) num_threads(threads)
	for (int i = 0; i < state.size; i++) {
		neigh = state.merges[i];
		// Only lower index performs merging, to avoid duplicate merges
		if (neigh > i && i == state.merges[neigh]) {
			assert(state.alive[i] && state.alive[neigh]);
			assert(getDistance(i, neigh) < clusteringOptions->params.radius);
			assert(state.clusters[state.indexes[neigh]].x == state.indexes[neigh]);
			// Mark cluster as "dead"
			numAlive--;
			state.alive[neigh] = false;
			state.weigths[i] += state.weigths[neigh];
			// Assign cluster, requires lookup to the index table due to re-sizing
			state.clusters[state.indexes[neigh]].x = state.indexes[i];
			state.clusters[state.indexes[neigh]].y = state.iteration;
		}
	}
	std::cout << "Total merged clusters: " << (state.numAlive - numAlive) << std::endl;
	state.numAlive = numAlive;
	return true;
}


bool ClusteringParallelCPU::computeMovements() {
	// Check if clustering was initialized
	if (!state.isInitialized()) {
		std::cout << "Error cannot compute movements, state is not initialized!" << std::endl;
		return false;
	}
	// Compute attraction between all points (skip the point itself . division by zero)
	float diffX, diffY, attraction;
	float2 pos, move;
	omp_set_dynamic(0);
	#pragma omp parallel for private(diffX, diffY, attraction, pos, move) num_threads(threads)
	for (int i = 0; i < state.size; i++) {
		if (!state.alive[i]) {
			continue;
		}
		pos = state.positions[i];
		move = { 0., 0. };
		// First half
		#pragma omp simd
		for (int j = 0; j < i; j++) {
			if (!state.alive[j]) {
				continue;
			}
			diffX = (state.positions[j].x - pos.x);
			diffY = (state.positions[j].y - pos.y);
			attraction = state.weigths[j] / ((diffX * diffX) + (diffY * diffY));
			move.x += (diffX * attraction);
			move.y += (diffY * attraction);
		}
		// Second half
		#pragma omp simd
		for (int j = i + 1; j < state.size; j++) {
			if (!state.alive[j]) {
				continue;
			}
			diffX = (state.positions[j].x - pos.x);
			diffY = (state.positions[j].y - pos.y);
			attraction = state.weigths[j] / ((diffX * diffX) + (diffY * diffY));
			move.x += (diffX * attraction);
			move.y += (diffY * attraction);
		}
		state.movements[i] = move;
	}
	// Shift original positions by the computed movements
	omp_set_dynamic(0);
	#pragma omp parallel for num_threads(threads)
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

bool ClusteringParallelCPU::countAlive() {
	// Check if we should resize
	assert(state.isInitialized() && partial != nullptr);
	if (!isStateResizable()) {
		std::cout << "Skipping resizing ..." << std::endl;
		return true;
	}
	std::cout << "Resizing points to size: " << state.numAlive << " from: " << state.size << std::endl;
	// Compute prefix scan for alive array
	const int work1 = state.size / threads;
	const int work2 = state.numAlive / threads;
	assert(work1 > 0 && work2 > 0);
	// Allocate temporary place holders
	float2* temp_positions = (float2*)malloc(state.numAlive * sizeof(float2));
	float* temp_weigths = (float*)malloc(state.numAlive * sizeof(float));
	int* temp_indexes = (int*)malloc(state.numAlive * sizeof(int));
	// Check allocation
	if (temp_positions == nullptr || temp_weigths == nullptr || temp_indexes == nullptr) {
		std::cout << "Allocation of memory failed, possibly out of memory!" << std::endl;
		throw std::bad_alloc();
	}
	// Shift all alive clusters in arrays to the front (0 to numAlive)
	omp_set_dynamic(0);
	#pragma omp parallel num_threads(threads)
	{	
		// ------  Compute partial sums  ------ 
		const int tid = omp_get_thread_num();
		const int a1 = (tid * work1), a2 = (tid * work2);
		const int b1 = ((tid == threads - 1) ? state.size : (a1 + work1)), b2 = ((tid == threads - 1) ? state.numAlive : (a2 + work2));
		int sum = 0;
		for (int i = a1; i < b1; i++) {
			sum += state.alive[i];
		}
		partial[tid] = sum;
		#pragma omp barrier
		// ------  Copy elements to temporary arrays  ------  
		sum = 0; // Use sum as offset
		// Compute the offset from previous sums
		for (int i = 0; i < tid; i++) {
			sum += partial[i];
		}
		// Skip empty chunks
		if (partial[tid] != 0) {
			// Shift all alive points in arrays to the front, based on the prefix scan
			for (int i = a1; i < b1; i++) {
				if (state.alive[i]) {
					assert(sum < state.numAlive);
					temp_positions[sum] = state.positions[i];
					temp_weigths[sum] = state.weigths[i];
					temp_indexes[sum] = state.indexes[i];
					sum++;
				}
			}
		}
		// ------  Copy temporary arrays to State  ------  
		#pragma omp barrier
		for (int i = a2; i < b2; i++) {
			state.alive[i] = true;
			state.positions[i] = temp_positions[i];
			state.weigths[i] = temp_weigths[i];
			state.indexes[i] = temp_indexes[i];
		}
	}
	// Release memory of temporary arrays
	free(temp_indexes);
	free(temp_positions);
	free(temp_weigths);
	state.allocate(state.numAlive);
	return true;
}


float4 ClusteringParallelCPU::findBorders() {
	std::cout << "*) Looking for Grid borders" << std::endl;
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
	const int chunk = state.size / omp_get_max_threads();
	// Compute the smallest coordinates
	omp_set_dynamic(0);
	#pragma omp parallel num_threads(threads)
	{
		const int tid = omp_get_thread_num();
		const int a = tid * chunk;
		const int b = (tid == omp_get_max_threads() - 1) ? state.numAlive : (a + chunk);
		float4 temp = float4{
			std::numeric_limits<float>::min(), std::numeric_limits<float>::min(),
			std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
		};
		// Find borders for each chunk
		for (int i = a; i < b; i++) {
			if (!state.alive[i]) {  // Skip dead point
				continue;
			}
			temp.x = std::max(temp.x, state.positions[i].x);
			temp.y = std::max(temp.y, state.positions[i].y);
			temp.z = std::min(temp.z, state.positions[i].x);
			temp.w = std::min(temp.w, state.positions[i].y);
		}
		// Determine coordinates between chunks
		#pragma omp critical
		{
			borders.x = std::max(borders.x, temp.x);
			borders.y = std::max(borders.y, temp.y);
			borders.z = std::min(borders.z, temp.z);
			borders.w = std::min(borders.w, temp.w);
		}
	}
	assert(borders.x != std::numeric_limits<float>::min() && borders.y != std::numeric_limits<float>::min());
	assert(borders.z != std::numeric_limits<float>::max() && borders.w != std::numeric_limits<float>::max());
	assert(0.f <= borders.z && borders.z < borders.x);
	assert(0.f <= borders.w && borders.w < borders.y);
	const auto result = since(start);
	record.bordersElapsed = result.count();
	std::cout << "*) Finished finding borders, elapsed=" << toMS(result) << "[ms]" << std::endl;
	return borders;
}
