#include "../../headers/clustering/gpu/host.cuh"
#include <iomanip>

// ------------------------------------------- Clustering Methods ------------------------------------------- 

void insertPointsCUDA(State stateGPU, Grid gridGPU, const int maxThreads, const float radiusFrac) {
	assert(stateGPU.isInitialized() && gridGPU.isInitialized() && !stateGPU.host && !gridGPU.host);
	// Compute number of blocks & threads needed
	const int blocks = std::max(1, static_cast<int>(ceil((float) stateGPU.size / (float) maxThreads)));
	const int threads = (blocks > 1) ? maxThreads : std::min(maxThreads, stateGPU.size);
	assert(blocks * threads >= stateGPU.size);
	
	// GpuTimer timer = GpuTimer();
	// timer.start();
	// Compute cell for each edge and increase bit counts
	std::cout << "Inserting " << stateGPU.numAlive << "/" << stateGPU.size << " clusters to grid!" << std::endl;
	std::cout << "(Blocks, threads): " << blocks << ", " << threads << std::endl;
	insertPointsKern<<<blocks, threads>>>(stateGPU, gridGPU, radiusFrac);
	CHECK_ERROR(cudaDeviceSynchronize());
	// Compute prefix scan on bin counts
	// prescanArrayRecursiveInt(gridGPU.binCounts, gridGPU.gridMap, gridGPU.size, 0, maxThreads);
	std::cout << "Finished prescan on gridMap & binCounts" << std::endl;
	
	// Sort array by prefixscan
	// sortPointsKern<<<blocks, threads>>>(stateGPU, gridGPU);


	// timer.stop();
	// std::cout << "4) Finished inserting clusters, elapsed=" << timer.elapsed() << " [ms]" << std::endl;
	// Check inserted points
	/*
	int* tmpCells = (int*)malloc(stateGPU.size * sizeof(int));
	int* tmpPointOrder = (int*)malloc(stateGPU.size * sizeof(int));
	assert(tmpCells != nullptr && tmpPointOrder != nullptr);
	CHECK_ERROR(cudaMemcpy(tmpCells, gridGPU.cells, stateGPU.size * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK_ERROR(cudaMemcpy(tmpPointOrder, gridGPU.pointOrder, stateGPU.size * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 100; i < 150; i++) {
		std::cout << "Cluster: " << i << " is at cell: " << tmpCells[i] << " order: " << tmpPointOrder[i] << std::endl;
	}
	free(tmpCells);
	*/
}


void computeMovementsCUDA(State& stateGPU, const int threads, const float4& borders) {
	std::cout << "Computing movements on cuda!" << std::endl;
}

void findClustersCUDA(State& stateGPU, Grid& gridGPU, const float radius2) {

}

void mergeClustersCUDA(State& stateGPU) {

}


float4 findBordersCUDA(State& stateGPU, const int maxThreads) {
	assert(bordersBlocks != nullptr);
	std::cout << "Finding borders for size: " << stateGPU.size << " levels: " << bordersBlocks->levels << std::endl;
	GpuTimer timer = GpuTimer();
	timer.start();
	// Compute number of blocks & threads needed
	int size = stateGPU.size;
	int blocks = std::max(1, static_cast<int>(ceil((float) size / (2.f * maxThreads))));
	int threads = getThreadsCount(blocks, size, maxThreads);
	unsigned int sharedMemSize = threads * sizeof(float4) * 2;
	std::cout << "Computing level: 0" << " blocks: " << blocks << " threads: " << threads << " size: " << size << std::endl;
	// Peform parallel reduction UP
	prescanF2<<<blocks, threads, sharedMemSize>>>(stateGPU.positions, bordersBlocks->blocks[0], stateGPU.alive, size, 0, 0);
	for (int i = 1; i < bordersBlocks->levels; i++) {
		size = blocks;
		blocks = std::max(1, static_cast<int>(ceil((float)size / (2.f * maxThreads))));
		threads = getThreadsCount(blocks, size, maxThreads);
		sharedMemSize = threads * sizeof(float4) * 2;
		std::cout << "Computing level: " << i << " blocks: " << blocks << " threads: " << threads << " size: " << size << std::endl;
 		// prescanF4<<<blocks, threads, sharedMemSize>>>(bordersBlocks->blocks[i-1], bordersBlocks->blocks[i], size, 0, 0);
	}
	CHECK_ERROR(cudaDeviceSynchronize());
	timer.stop();
	std::cout << "Time taken: " << timer.elapsed() << " [ms]" << std::endl;
	// Get the final result and copy it to CPU
	float4 borders;
	CHECK_ERROR(cudaMemcpy(&borders, bordersBlocks->blocks[bordersBlocks->levels-1], sizeof(float4), cudaMemcpyDeviceToHost));
	return borders;
}

// ------------------------------------------- Recursive scan -------------------------------------------
/*
void prescanArrayRecursiveInt(const int* inArray, int* outArray, int numElements, int level, const int maxThreads) {
	// Compute number of blocks & threads needed
	const int blocks = std::max(1, static_cast<int>(ceil((float)numElements / (2.f * maxThreads))));
	const int threads = getThreadsCount(blocks, numElements, maxThreads);
	const unsigned int sharedMemSize = threads * sizeof(float4) * 2;
	std::cout << "Prefix scan level: " << level << " blocks: " << blocks << " threads: " << threads << " size: " << numElements << std::endl;
	assert(level <= gridLevels);
	// execute the scan
	if (blocks > 1) {
		prescanInt<<<blocks, threads, sharedMemSize>>>(inArray, outArray, partialGrid[level], numElements, 0, 0);
		// After scanning all the sub-blocks, we are mostly done.  But now we 
		// need to take all of the last values of the sub-blocks and scan those.  
		// This will give us a new value that must be added to each block to 
		// get the final results.
		
		// recursive (CPU) call
		prescanArrayRecursiveInt(partialGrid[level], partialGrid[level], blocks, level + 1, maxThreads);
		addInt<<<blocks, threads>>>(outArray, partialGrid[level], numElements, 0, 0);
	} else if (isPowerOfTwo(numElements)) {
		assert(numElements == 2 * threads);
		prescanInt<<<blocks, threads, sharedMemSize>>>(inArray, outArray, nullptr, threads * 2, 0, 0);
	} else {
		prescanInt<<<blocks, threads, sharedMemSize>>>(inArray, outArray, nullptr, numElements, 0, 0);
	}
}
*/


void testPrefix() {




}









