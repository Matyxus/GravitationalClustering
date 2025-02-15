#include "../../headers/clustering/gpu/alloc.cuh"
#include <iomanip>

// --------------------------------------------- Init ---------------------------------------------

bool initializeCuda() {
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

// --------------------------------------------- Blocks ---------------------------------------------

template <typename T> 
ReductionBlock<T>::ReductionBlock(const int totalSize, const int threads, const bool allocateLast, const bool allocateHelper) :
	levels(computeLevels(totalSize, threads, allocateLast)) 
{
	printf("Allocating blocks for size=%d, threads=%d, (last, helper)=(%d,%d)\n", totalSize, threads, allocateLast, allocateHelper);
	assert(totalSize > 1 && threads >= 32);
	blocks = static_cast<T**>(std::malloc(levels * sizeof(T*)));
	assert(blocks != nullptr);
	const float work = 2.0f * threads; // Total amount of work per threads in block
	int size = totalSize, level = 0;
	if (allocateHelper) {
		CHECK_ERROR(cudaMalloc((void**)&helper, totalSize * sizeof(T)));
	}
	while (size > 1) {
		int numBlocks = std::max(1, static_cast<int>(ceil((float)size / work)));
		// No need to allocate last level here for total sum
		if (numBlocks > 1 || allocateLast) {
			CHECK_ERROR(cudaMalloc((void**)&blocks[level], numBlocks * sizeof(T)));
			level++;
		}
		size = numBlocks;
	}
}

template <typename T> 
ReductionBlock<T>::~ReductionBlock() {
	if (blocks != nullptr) {
		assert(levels > 0);
		for (int i = 0; i < levels; i++) {
			CHECK_ERROR(cudaFree(blocks[i]));
		}
		free(blocks);
		blocks = nullptr;
	}
	if (helper != nullptr) {
		CHECK_ERROR(cudaFree(helper));
		helper = nullptr;
	}
}

template <typename T> 
int ReductionBlock<T>::computeLevels(const int totalSize, const int threads, const bool allocateLast) {
	const float work = 2.0f * threads; // Total amount of work per threads in block
	int size = totalSize;
	if (size <= 1) {
		std::cout << "Error, compute reduction levels for size: " << size << std::endl;
		return -1;
	}
	int level = 0;
	int numBlocks;
	// Compute the total number of levels needed
	while (size > 1) {
		numBlocks = std::max(1, static_cast<int>(ceil((float)size / work)));
		if (numBlocks > 1 || allocateLast) {
			printf("Level=%d has %d=blocks\n", level, numBlocks);
			level++;
		}
		size = numBlocks;
	}
	return level;
}

// Possible data that can be used in template.
template struct ReductionBlock<int>;
template struct ReductionBlock<float4>;
// Declared global variables for blocks
struct ReductionBlock<float4>* bordersBlocks = nullptr;
struct ReductionBlock<int>* aliveBlocks = nullptr;
struct ReductionBlock<int>* gridBlocks = nullptr;


// --------------------------------------------- Allocate ---------------------------------------------


void allocateStateCUDA(State& stateCPU, State& stateGPU) {
	std::cout << "Allocating GPU state" << std::endl;
	assert(!stateGPU.isInitialized() && stateCPU.isInitialized());
	assert(!stateGPU.host && stateCPU.host);
	// ----------------- Initialize State -----------------
	stateGPU.size = stateCPU.size;
	stateGPU.numAlive = stateCPU.numAlive;
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.positions, stateGPU.size * sizeof(float2)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.movements, stateGPU.size * sizeof(float2)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.weigths, stateGPU.size * sizeof(float)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.clusters, stateGPU.size * sizeof(int2)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.indexes, stateGPU.size * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.merges, stateGPU.size * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&stateGPU.alive, stateGPU.size * sizeof(bool)));
	// Copy data
	CHECK_ERROR(cudaMemcpy(stateGPU.positions, stateCPU.positions, stateGPU.size * sizeof(float2), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.movements, stateCPU.movements, stateGPU.size * sizeof(float2), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.weigths, stateCPU.weigths, stateGPU.size * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.clusters, stateCPU.clusters, stateGPU.size * sizeof(int2), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.indexes, stateCPU.indexes, stateGPU.size * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.merges, stateCPU.merges, stateGPU.size * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(stateGPU.alive, stateCPU.alive, stateGPU.size * sizeof(bool), cudaMemcpyHostToDevice));
}


void allocateGridCUDA(Grid& gridGPU, const float4 borders, const int numClusters, const float radius) {
	assert(!gridGPU.isInitialized() && !gridGPU.host);
	assert(borders.x != std::numeric_limits<float>::min() && borders.y != std::numeric_limits<float>::min());
	assert(borders.z != std::numeric_limits<float>::max() && borders.w != std::numeric_limits<float>::max());
	assert(0 <= borders.z && borders.z < borders.x);
	assert(0 <= borders.w && borders.w < borders.y);
	// Assign dimensions
	gridGPU.borders = borders;
	gridGPU.rows = static_cast<int>(std::ceil((gridGPU.borders.x - gridGPU.borders.z) / radius));
	gridGPU.cols = static_cast<int>(std::ceil((gridGPU.borders.y - gridGPU.borders.w) / radius));
	gridGPU.size = (gridGPU.rows * gridGPU.cols);
	assert(gridGPU.rows > 0 && gridGPU.cols > 0);
	std::cout << "Grid limits are: ((" << gridGPU.borders.z << "," << gridGPU.borders.w << "), (" << gridGPU.borders.x << "," << gridGPU.borders.y << "))" << std::endl;
	std::cout << "New grid dimensions are: (" << gridGPU.rows << "," << gridGPU.cols << ") -> " << gridGPU.size << std::endl;
	// Allocate array's
	CHECK_ERROR(cudaMalloc((void**)&gridGPU.cells, numClusters * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&gridGPU.sortedCells, numClusters * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&gridGPU.pointOrder, numClusters * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&gridGPU.gridMap, gridGPU.size * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**)&gridGPU.binCounts, gridGPU.size * sizeof(int)));
	// Set to zero values
	CHECK_ERROR(cudaMemset(gridGPU.gridMap, 0, gridGPU.size * sizeof(int)));
	CHECK_ERROR(cudaMemset(gridGPU.binCounts, 0, gridGPU.size * sizeof(int)));
	// Assign neighbours
	gridGPU.neighbours[2] = gridGPU.rows;
	gridGPU.neighbours[3] = gridGPU.rows + 1;
	gridGPU.neighbours[4] = gridGPU.rows - 1;
	gridGPU.neighbours[5] = -gridGPU.rows;
	gridGPU.neighbours[6] = -gridGPU.rows + 1;
	gridGPU.neighbours[7] = -gridGPU.rows - 1;
	// Check allocation
	assert(gridGPU.isInitialized());
}

void allocateStateHelpers(const int numClusters, const int threads) {
	std::cout << "----- Allocating StateGPU helpers -----" << std::endl;
	assert(bordersBlocks == nullptr && aliveBlocks == nullptr);
	bordersBlocks = new struct ReductionBlock<float4>(numClusters, threads, true, false);
	aliveBlocks = new struct ReductionBlock<int>(numClusters, threads, false, true);
	assert(bordersBlocks != nullptr && aliveBlocks != nullptr);
	CHECK_ERROR(cudaMemset(aliveBlocks->helper, 0, numClusters * sizeof(int)));
	std::cout << "--------- Finished ---------" << std::endl;
}

void allocateGridHelpers(const int gridSize, const int threads) {
	std::cout << "----- Allocating GridGPU helpers -----" << std::endl;
	assert(gridBlocks == nullptr);
	gridBlocks = new struct ReductionBlock<int>(gridSize, threads, false, false);
	assert(gridBlocks != nullptr);
	std::cout << "--------- Finished ---------" << std::endl;
}

// --------------------------------------------- Free ---------------------------------------------

void freeStateCUDA(State& stateGPU) {
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

void freeGridCUDA(Grid& gridGPU) {
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
}

void freeHelpers() {
	std::cout << "Freeing GPU helper vars" << std::endl;
	if (bordersBlocks != nullptr) {
		delete bordersBlocks;
		bordersBlocks = nullptr;
	}
	if (aliveBlocks != nullptr) {
		delete aliveBlocks;
		aliveBlocks = nullptr;
	}
	if (gridBlocks != nullptr) {
		delete gridBlocks;
		gridBlocks = nullptr;
	}
}

// --------------------------------------------- Utils ---------------------------------------------


void resetGridCUDA(Grid& gridGPU, const int numClusters) {
	assert(gridGPU.isInitialized() && !gridGPU.host && numClusters > 1);
	// Zero out values of grid arrays
	GpuTimer timer = GpuTimer();
	timer.start();
	CHECK_ERROR(cudaMemset(gridGPU.cells, 0, numClusters * sizeof(int)));
	CHECK_ERROR(cudaMemset(gridGPU.sortedCells, 0, numClusters * sizeof(int)));
	CHECK_ERROR(cudaMemset(gridGPU.pointOrder, 0, numClusters * sizeof(int)));
	CHECK_ERROR(cudaMemset(gridGPU.gridMap, 0, gridGPU.size * sizeof(int)));
	CHECK_ERROR(cudaMemset(gridGPU.binCounts, 0, gridGPU.size * sizeof(int)));
	timer.stop();
	std::cout << "2) Finished re-sizing Grid: " << timer.elapsed() << " [ms]" << std::endl;
}

void copyToHost(State& stateCPU, State& stateGPU) {
	assert(stateGPU.isInitialized() && stateCPU.isInitialized());
	assert(stateGPU.size <= stateCPU.size && !stateGPU.host && stateCPU.host);
	stateCPU.size = stateGPU.size;
	stateCPU.alive = stateGPU.alive;
	CHECK_ERROR(cudaMemcpy(stateCPU.positions, stateGPU.positions, stateGPU.size * sizeof(float2), cudaMemcpyDeviceToHost));
	CHECK_ERROR(cudaMemcpy(stateCPU.movements, stateGPU.movements, stateGPU.size * sizeof(float2), cudaMemcpyDeviceToHost));
	CHECK_ERROR(cudaMemcpy(stateCPU.weigths, stateGPU.weigths, stateGPU.size * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_ERROR(cudaMemcpy(stateCPU.clusters, stateGPU.clusters, stateGPU.size * sizeof(int2), cudaMemcpyDeviceToHost));
	CHECK_ERROR(cudaMemcpy(stateCPU.indexes, stateGPU.indexes, stateGPU.size * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK_ERROR(cudaMemcpy(stateCPU.merges, stateGPU.merges, stateGPU.size * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK_ERROR(cudaMemcpy(stateCPU.alive, stateGPU.alive, stateGPU.size * sizeof(bool), cudaMemcpyDeviceToHost));
}


