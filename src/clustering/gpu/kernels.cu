#include "../../headers/clustering/gpu/kernels.cuh"



__global__ void insertPointsKern(State stateGPU, Grid gridGPU, const float radiusFrac) {
	const int thid = threadIdx.x + blockIdx.x * blockDim.x;
	if (thid >= stateGPU.size || !stateGPU.alive[thid]) {
		return;
	}
	const float2 pos = stateGPU.positions[thid];
	const int x = static_cast<int>((pos.x - gridGPU.borders.z) * radiusFrac);
	const int y = static_cast<int>((pos.y - gridGPU.borders.w) * radiusFrac);
	const int cell = x + y * gridGPU.rows;
	assert(x >= 0 && y >= 0);
	assert(0 <= cell && cell < gridGPU.size);
	gridGPU.cells[thid] = cell;
	gridGPU.pointOrder[thid] = atomicAdd(&(gridGPU.binCounts[cell]), 1);
}

__global__ void sortPointsKern(State stateGPU, Grid gridGPU) {
	const int thid = threadIdx.x + blockIdx.x * blockDim.x;
	if (thid >= stateGPU.size || !stateGPU.alive[thid]) {
		return;
	}
	assert(0 <= gridGPU.gridMap[gridGPU.cells[thid]] + gridGPU.pointOrder[thid] && gridGPU.gridMap[gridGPU.cells[thid]] + gridGPU.pointOrder[thid] < gridGPU.size);
	gridGPU.sortedCells[gridGPU.gridMap[gridGPU.cells[thid]] + gridGPU.pointOrder[thid]] = thid;
}

// ------------- Movements ------------- 
/*
__device__ void loadChunk(float4 *cache, State stateGPU, const int chunkSize, int ai, int bi, int bankOffsetA, int bankOffsetB) {
	// First element
	if (ai + chunkSize < stateGPU.size && stateGPU.alive[ai + chunkSize]) {
		const float2 first = stateGPU.positions[ai + chunkSize];
		cache[ai + bankOffsetA] = float4{first.x, first.y, stateGPU.weigths[ai + chunkSize],  1.f};
	} else { // Out of bounds or dead 
		cache[ai + bankOffsetA] = float4{0.f, 0.f, 0.f, 0.f};
	}
	// Second element
	if (bi + chunkSize < stateGPU.size && stateGPU.alive[bi + chunkSize]) {
		const float2 second = stateGPU.positions[bi + chunkSize];
		cache[bi + bankOffsetB] = float4{ second.x, second.y, stateGPU.weigths[bi + chunkSize], 1.f};
	} else { // Out of bounds or dead
		cache[bi + bankOffsetB] = float4{ 0.f, 0.f, 0.f, 0.f };
	}
	__syncthreads();
}

__device__ void upScanF4(float4* cache) {
	const unsigned int thid = threadIdx.x;
	unsigned int ai, bi;
	unsigned int offset = 1;
	// Up-Sweep (Reduce), Block B
	for (unsigned int d = blockDim.x; d > 0; d >>= 1) {
		__syncthreads();
		if (thid < d) {
			ai = 2 * offset * thid + offset - 1;
			bi = ai + offset;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			cache[bi] += cache[ai];
		}
		offset *= 2;
	}
}

__device__ void saveSumInt(float4* cache, int* out, const int blockIndex) {
	if (threadIdx.x == 0) {
		int index = (2 * blockDim.x) - 1;
		index += CONFLICT_FREE_OFFSET(index);
		// Store total sum to upper level
		if (out != nullptr) {
			out[blockIndex] = cache[index];
		}
		// Zero out last
		cache[index] = 0;
	}
}


__global__ void computeMovementsKern(State stateGPU, float2 *movements, const int chunkSize, const int chunks, const float radius2) {
	extern __shared__ float4 cache[]; // {x, y, weigth, attraction}
	__shared__ float2 pos; // Position (constant) -> (x, y)
	__shared__ float2 total;
	const int thid = 2 * blockDim.x * blockIdx.x + threadIdx.x;
	if (threadIdx.x == 0) {
		pos = stateGPU.positions[blockIdx.x];
		total = float2{ 0.f, 0.f };
	}
	int ai, bi, bankOffsetA, bankOffsetB;
	float4 target;
	float2 diff;
	ai = threadIdx.x;
	bi = threadIdx.x + blockDim.x;
	bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	__syncthreads();
	// for each chunk in chunks
	for (int i = 0; i < chunks; i++) {
		// Load_chunk + chek that element != pos
		loadChunk(cache, stateGPU, chunkSize, ai, bi, bankOffsetA, bankOffsetB);
		// Compute movements
		target = cache[ai + bankOffsetA];
		if (target.z != 0.f) { // Check for alive
			diff = float2{ target.x - pos.x, target.y - pos.y };
			target.w = target.z / (diff.x * diff.x + diff.y + diff.y);
			target.x = diff.x * target.w;
			target.y = diff.y * target.w;
		}
		target = cache[bi + bankOffsetB];
		if (target.z != 0.f) { // Check for alive
			diff = float2{ target.x - pos.x, target.y - pos.y };
			target.w = target.z / (diff.x * diff.x + diff.y + diff.y);
			target.x = diff.x * target.w;
			target.y = diff.y * target.w;
		}
		// Prefix sum
		upScanF4(cache);
		// Add total
		if (threadIdx.x == 0) {
			total.x += cache[2 * blockDim.x - 1].x;
			total.y += cache[2 * blockDim.x - 1].y;
		}
	}
	// Store total
	if (threadIdx.x == 0) {
		movements[blockIdx.x] = total;
	}
}
*/





