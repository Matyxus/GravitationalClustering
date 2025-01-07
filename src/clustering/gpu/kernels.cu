#include "../../headers/clustering/gpu/kernels.cuh"



__global__ void insertPointsKern(State stateGPU, Grid gridGPU, const float radius) {
	const int thid = threadIdx.x + blockIdx.x * blockDim.x;
	if (thid >= stateGPU.numEdges || !stateGPU.alive[thid]) {
		return;
	}
	const float2 pos = stateGPU.positions[thid];
	const int x = static_cast<int>((pos.x - gridGPU.borders.z) / radius);
	const int y = static_cast<int>((pos.y - gridGPU.borders.w) / radius);
	const int cell = x + y * gridGPU.rows;
	assert(x >= 0 && y >= 0);
	assert(0 <= cell && cell < gridGPU.size);
	gridGPU.cells[cell] = thid;
	gridGPU.pointOrder[thid] = atomicAdd(&gridGPU.binCounts[cell], 1);
}


// ------------- Movements ------------- 

__device__ void loadChunk(float4 *cache, State stateGPU, const int chunkSize, int& ai, int& bi, int& bankOffsetA, int& bankOffsetB) {
	ai = threadIdx.x;
	bi = threadIdx.x + blockDim.x;
	bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	// First element
	if (ai + chunkSize < stateGPU.numEdges && stateGPU.alive[ai + chunkSize]) {
		const float2 first = stateGPU.positions[ai + chunkSize];
		cache[ai + bankOffsetA] = float4{first.x, first.y, 1.f, stateGPU.weigths[ai + chunkSize]};
	} else { // Out of bounds or dead 
		cache[ai + bankOffsetA] = float4{0.f, 0.f, 0.f, 0.f};
	}
	// Second element
	if (bi + chunkSize < stateGPU.numEdges && stateGPU.alive[bi + chunkSize]) {
		const float2 second = stateGPU.positions[bi + chunkSize];
		cache[bi + bankOffsetB] = float4{ second.x, second.y, 1.f, stateGPU.weigths[bi + chunkSize] };
	} else { // Out of bounds or dead
		cache[bi + bankOffsetB] = float4{ 0.f, 0.f, 0.f, 0.f };
	}
	__syncthreads();
}


__global__ void computeMovementsKern(State stateGPU, const int chunkSize, const float radius2) {
	extern __shared__ float4 cache[]; // {x, y, weigth, attraction}
	__shared__ float2 pos; // Position (constant)
	const int thid = threadIdx.x;
	if (thid == 0) {
		pos = stateGPU.positions[blockIdx.x];
	}
	int ai, bi, bankOffsetA, bankOffsetB;
	__syncthreads();
	// for each chunk in chunks
	//	load_chunk + chek that element != pos
	//	compute movements
	//  sum

	
}







