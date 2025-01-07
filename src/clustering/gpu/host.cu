#pragma once
#include "../../headers/clustering/gpu/host.cuh"
#include <iomanip>


// ------------------------------------------- Init + Allocs  ------------------------------------------- 

bool initializeCuda() {
	// --------------- Check CUDA device --------------- 
	int count = 0;
	int i = 0;
	cudaError_t err = cudaGetDeviceCount(&count);
	if (err == cudaErrorInsufficientDriver) {
		std::cerr << "Error: CUDA driver not installed" << std::endl;
		return false;
	}
	else if (err == cudaErrorNoDevice || count == 0) {
		std::cerr << "Error: No CUDA device found" << std::endl;
		return false;
	}
	for (; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}
	if (i == count) {
		std::cerr << "Error: No CUDA device found" << std::endl;
		return false;
	}
	cudaSetDevice(i);
	std::cout << "CUDA initialized" << std::endl;
	cudaDeviceProp p;
	cudaGetDeviceProperties(&p, i);
	std::cout << "-------------- CUDA --------------" << std::endl;
	std::cout << "Devices:            " << count << std::endl;
	std::cout << "Device:             " << i << std::endl;
	std::cout << "Name:               " << p.name << std::endl;
	std::cout << "Revision:           " << p.major << "." << p.minor << std::endl;
	std::cout << "Global Mem:         " << std::setprecision(2) << (p.totalGlobalMem / (double)(1024 * 1024 * 1024)) << "[GB]" << std::endl;
	std::cout << "Shared/Blk:         " << ((int)(p.sharedMemPerBlock / 1024)) << "[KB]" << std::endl;
	std::cout << "Regs/Blk:           " << p.regsPerBlock << std::endl;
	std::cout << "Warp Size:          " << p.warpSize << std::endl;
	std::cout << "Mem Pitch:          " << ((int)(p.memPitch / 1024)) << "[KB]" << std::endl;
	std::cout << "Thrds/Blk:          " << p.maxThreadsPerBlock << std::endl;
	std::cout << "Const Mem:          " << ((int)(p.totalConstMem / 1024)) << "[KB]" << std::endl;
	std::cout << "Memory Clock:       " << std::setprecision(1) << (p.memoryClockRate / 1024) << "[MHz]" << std::endl;
	std::cout << "Memory Bus Width:   " << p.memoryBusWidth << "[B]" << std::endl;
	std::cout << "Peak Mem Bandwidth: " << std::fixed << ((double)(2.0 * p.memoryClockRate * (p.memoryBusWidth / 8) / 1.0e6)) << "[GB/s]" << std::endl;
	std::cout << "Concurrent kernels: " << (p.concurrentKernels ? "yes" : "no") << std::endl;
	std::cout << "Concurrent computation/communication: " << (p.deviceOverlap ? "yes" : "no") << std::endl;
	// std::cout << "Clock Rate: " << p.clockRate << std::endl; depracted
	std::cout << "-------------- CUDA --------------" << std::endl;
	return true;
}

bool allocateStateCUDA(State &stateCPU, State& stateGPU) {
	std::cout << "Allocating GPU state" << std::endl;
	assert(!stateGPU.isInitialized() && stateCPU.isInitialized());
	// ----------------- Initialize State -----------------
	stateGPU.numEdges = stateCPU.numEdges;
	stateGPU.numAlive = stateCPU.numAlive;
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.positions, stateGPU.numEdges * sizeof(float2)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.movements, stateGPU.numEdges * sizeof(float2)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.weigths, stateGPU.numEdges * sizeof(float)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.clusters, stateGPU.numEdges * sizeof(int2)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.indexes, stateGPU.numEdges * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.merges, stateGPU.numEdges * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.alive, stateGPU.numEdges * sizeof(bool)));
	// Copy data
	CHECK_ERROR(cudaMemcpy(stateGPU.positions, stateCPU.positions, stateGPU.numEdges * sizeof(float2), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.movements, stateCPU.movements, stateGPU.numEdges * sizeof(float2), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.weigths, stateCPU.weigths, stateGPU.numEdges * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.clusters, stateCPU.clusters, stateGPU.numEdges * sizeof(int2), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.indexes, stateCPU.indexes, stateGPU.numEdges * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.merges, stateCPU.merges, stateGPU.numEdges * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.alive, stateCPU.alive, stateGPU.numEdges * sizeof(bool), cudaMemcpyHostToDevice));
	return true;
}



// ------------------------------------------- Methods ------------------------------------------- 

bool generateGridCUDA(Grid& gridGPU, const float4 borders, const int numEdges, const float radius) {
	assert(!gridGPU.isInitialized());
	assert(borders.x != std::numeric_limits<float>::min() && borders.y != std::numeric_limits<float>::min());
	assert(borders.z != std::numeric_limits<float>::max() && borders.w != std::numeric_limits<float>::max());
	// Assign dimensions
	gridGPU.borders = borders;
	assert(0 <= gridGPU.borders.z && gridGPU.borders.z < gridGPU.borders.x);
	assert(0 <= gridGPU.borders.w && gridGPU.borders.w < gridGPU.borders.y);
	gridGPU.rows = static_cast<int>(std::ceil((gridGPU.borders.x - gridGPU.borders.z) / radius));
	gridGPU.cols = static_cast<int>(std::ceil((gridGPU.borders.y - gridGPU.borders.w) / radius));
	gridGPU.size = (gridGPU.rows * gridGPU.cols);
	assert(gridGPU.rows > 0 && gridGPU.cols > 0);
	std::cout << "Grid limits are: ((" << gridGPU.borders.z << "," << gridGPU.borders.w << "), (" << gridGPU.borders.x << "," << gridGPU.borders.y << "))" << std::endl;
	std::cout << "New grid dimensions are: (" << gridGPU.rows << "," << gridGPU.cols << ") -> " << gridGPU.size << std::endl;
	// Allocate array's
	CHECK_ERROR(cudaMalloc((void**)&gridGPU.cells, numEdges * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&gridGPU.sortedCells, numEdges * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&gridGPU.pointOrder, numEdges * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&gridGPU.gridMap, gridGPU.size * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&gridGPU.binCounts, gridGPU.size * sizeof(int)));
	// Set to zero values
	CHECK_ERROR(cudaMemset(gridGPU.gridMap, 0, gridGPU.size * sizeof(int)));
	CHECK_ERROR(cudaMemset(gridGPU.binCounts, 0, gridGPU.size * sizeof(int)));
	// Check allocation
	assert(gridGPU.isInitialized());
	// Assign neighbours
	gridGPU.neighbours[2] = gridGPU.rows;
	gridGPU.neighbours[3] = gridGPU.rows + 1;
	gridGPU.neighbours[4] = gridGPU.rows - 1;
	gridGPU.neighbours[5] = -gridGPU.rows;
	gridGPU.neighbours[6] = -gridGPU.rows + 1;
	gridGPU.neighbours[7] = -gridGPU.rows - 1;
	return true;
}


bool insertPointsCUDA(State& stateGPU, Grid& gridGPU, const int threads, const float radius) {
	std::cout << "Inserting " << stateGPU.numAlive << " points to grid" << std::endl;
	const int blocks = std::max(1, static_cast<int>(ceil(stateGPU.numEdges / (float) threads)));
	assert(blocks * threads >= stateGPU.numEdges);
	std::cout << "Blocks, threads: " << blocks << ", " << threads << std::endl;
	GpuTimer timer = GpuTimer();
	timer.Start();
	// Compute cell for each edge and increase bit counts
	insertPointsKern<<<blocks, threads>>>(stateGPU, gridGPU, radius);
	// Compute prefix scan on bin counts
	
	// Sort array by prefixscan
	timer.Stop();
	std::cout << "Time taken: " << timer.Elapsed() << " [ms]" << std::endl;
	return true;
}

bool computeMovementsCUDA(State& stateGPU, const int threads, const float radius2) {
	std::cout << "Computeing movements on cuda!" << std::endl;




	return true;
}




float4 findBordersCUDA(float2* positions, bool* alive, float4 **partial, struct Level *borderLevels, const int levels, int size) {
	std::cout << "Getting borderds for size: " << size << " levels: " << levels << std::endl;
	assert(levels >= 1);
	struct Level* current = &borderLevels[0];
	assert(current != nullptr);
	std::cout << "Computing level: 0" << " blocks: " << current->blocks << " threads: " << current->threads << std::endl;
	std::cout << "Size: " << size << std::endl;
	GpuTimer timer = GpuTimer();
	timer.Start();
	prescanF2<<<current->blocks, current->threads, current->sharedMem>>>(positions, partial[0], alive, size, 0, 0);
	for (int i = 1; i < levels; i++) {
		size = current->blocks;
		current = &borderLevels[i];
		assert(current != nullptr);
		std::cout << "Computing level: " << i << " blocks: " << current->blocks << " threads: " << current->threads << std::endl;
		std::cout << "Size: " << size << std::endl;
 		prescanF4<<<current->blocks, current->threads, current->sharedMem>>>(partial[i-1], partial[i], size, 0, 0);
	}
	timer.Stop();
	std::cout << "Time taken: " << timer.Elapsed() << " [ms]" << std::endl;
	// Get the final result and copy it to CPU
	float4 borders;
	CHECK_ERROR(cudaDeviceSynchronize());
	CHECK_ERROR(cudaMemcpy(&borders, partial[levels-1], sizeof(float4), cudaMemcpyDeviceToHost));
	return borders;
}


