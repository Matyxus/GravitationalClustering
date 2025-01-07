#include "../../headers/clustering/gpu/prefix.cuh"

// -------------------------------- Load Shared Mem --------------------------------

// -------- Borders --------

__device__ void loadSharedF2(
    float4* cache, const float2* input, const bool* alive, const int size, const int shift,
    int& ai, int& bi, int& bankOffsetA, int& bankOffsetB
)
{
    ai = threadIdx.x;
    bi = threadIdx.x + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    // We have to put float2 into float4 by duplicating it,
    // This way we can run min/max on both coordinates
    assert(shift < size);
    const float2 first = (ai + shift < size && alive[ai + shift]) ? input[ai + shift] : input[0];
    cache[ai + bankOffsetA] = float4{ first.x, first.y, first.x, first.y };
    if (bi + shift < size && alive[bi + shift]) {
        const float2 second = input[bi + shift];
        cache[bi + bankOffsetB] = float4{ second.x, second.y, second.x, second.y };
    } else { // Duplicate the first loaded value again, cannot use default values
        cache[bi + bankOffsetB] = cache[ai + bankOffsetA];
    }
}


__device__ void loadSharedF4(
    float4* cache, const float4* input, const int size, const int shift,
    int& ai, int& bi, int& bankOffsetA, int& bankOffsetB
)
{
    ai = threadIdx.x;
    bi = threadIdx.x + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    // TODO load (inf, inf, inf, inf) boarderds
    assert(shift < size);
    cache[ai + bankOffsetA] = ((ai + shift < size) ? input[ai + shift] : input[shift]);
    cache[bi + bankOffsetB] = ((bi + shift < size) ? input[bi + shift] : cache[ai + bankOffsetA]);
}

// -------- Alive & Grid --------

__device__ void loadSharedInt(
    int* cache, const int* input, const int size, const int shift,
    int& ai, int& bi, int& bankOffsetA, int& bankOffsetB
)
{
    ai = threadIdx.x;
    bi = threadIdx.x + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    cache[ai + bankOffsetA] = ((ai + shift < size) ? input[ai + shift] : 0);
    cache[bi + bankOffsetB] = ((bi + shift < size) ? input[bi + shift] : 0);
}


__device__ void loadSharedBool(
    int* cache, const bool* input, const int size, const int shift,
    int& ai, int& bi, int& bankOffsetA, int& bankOffsetB
)
{
    ai = threadIdx.x;
    bi = threadIdx.x + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    cache[ai + bankOffsetA] = ((ai + shift < size) ? static_cast<int>(input[ai + shift]) : 0);
    cache[bi + bankOffsetB] = ((bi + shift < size) ? static_cast<int>(input[bi + shift]) : 0);
}


// -------------------------------- Store Shared Mem --------------------------------


__device__ void storeSharedInt(
    int* out, const int* cache, const int size, const int shift,
    int ai, int bi, int bankOffsetA, int bankOffsetB
)
{
    __syncthreads();
    out[ai + shift] = cache[ai + bankOffsetA];
    if (bi + shift < size) {
        out[bi + shift] = cache[bi + bankOffsetB];
    }
}



// -------------------------------- Save Sum --------------------------------

__device__ void saveSumInt(int* cache, int* out, const int blockIndex) {
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

__device__ void saveSumF4(float4* cache, float4* out, const int blockIndex) {
    if (threadIdx.x == 0) {
        int index = (2 * blockDim.x) - 1;
        index += CONFLICT_FREE_OFFSET(index);
        assert(out != nullptr);
        // Store min/max to upper level
        out[blockIndex] = cache[index];
        // printf("Tid %d saving (%.1f, %.1f, %.1f, %.1f) at %d\n", threadIdx.x, cache[index].x, cache[index].y, cache[index].z, cache[index].w, blockIndex);
        // No need to zero last, no down reduce for finding borders
    }
}


// -------------------------------- Scan UP --------------------------------

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
            cache[bi].x = std::max(cache[bi].x, cache[ai].x);
            cache[bi].y = std::max(cache[bi].y, cache[ai].y);
            cache[bi].z = std::min(cache[bi].z, cache[ai].z);
            cache[bi].w = std::min(cache[bi].w, cache[ai].w);
        }
        offset *= 2;
    }
}


__device__ unsigned int upScanInt(int* cache) {
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
    return offset;
}


// -------------------------------- Scan Down --------------------------------

// Only prefix sum is peforming down scan

__device__ void downScanInt(int* cache, unsigned int offset) {
    const unsigned int thid = threadIdx.x;
    unsigned int ai, bi;
    // Down-Sweep (Reduce), block D
    for (unsigned int d = 1; d <= blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            ai = 2 * offset * thid + offset - 1;
            bi = ai + offset;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int t = cache[ai];
            cache[ai] = cache[bi];
            cache[bi] += t;
        }
    }
}


// -------------------------------- Add kernel --------------------------------

__global__ void addInt(int* out, const int* sums, const int size, const int blockOffset, const int baseIndex) {
    __shared__ int sum;
    if (threadIdx.x == 0) {
        sum = sums[blockIdx.x + blockOffset];
    }
    const unsigned int address = 2 * blockDim.x * blockIdx.x + baseIndex + threadIdx.x;
    __syncthreads();
    // note two adds per thread
    out[address] += sum;
    out[address + blockDim.x] += (threadIdx.x + blockDim.x < size) * sum;
}

// -------------------------------- Scan Block --------------------------------

// Borders
__device__ void prescanBlockF4(float4* cache, float4* out, const int blockIndex) {
    upScanF4(cache);
    saveSumF4(cache, out, ((blockIndex == 0) ? blockIdx.x : blockIndex));
    // No down sweep
}


// Grid + Alive
__device__ void prescanBlockInt(int* cache, int* out, const int blockIndex) {
    const unsigned int offset = upScanInt(cache);
    saveSumInt(cache, out, ((blockIndex == 0) ? blockIdx.x : blockIndex));
    downScanInt(cache, offset);
}

// -------------------------------- Scan Kernels --------------------------------

// ----------------- Borders-----------------

__global__ void prescanF2(
    const float2* input, float4* out, const bool* alive,
    const int size, const int blockIndex, const int baseIndex
) {
    extern __shared__ float4 cacheF4[];
    int ai, bi, bankOffsetA, bankOffsetB;
    const int shift = ((baseIndex == 0) ? (2 * blockDim.x * blockIdx.x) : baseIndex);
    loadSharedF2(cacheF4, input, alive, size, shift, ai, bi, bankOffsetA, bankOffsetB);
    prescanBlockF4(cacheF4, out, blockIndex);
    // No need to store partial min/maxes
}

__global__ void prescanF4(
    const float4* input, float4* out, const int size,
    const int blockIndex, const int baseIndex
) {
    extern __shared__ float4 cacheF4_2[];
    int ai, bi, bankOffsetA, bankOffsetB;
    const int shift = ((baseIndex == 0) ? (2 * blockDim.x * blockIdx.x) : baseIndex);
    loadSharedF4(cacheF4_2, input, size, shift, ai, bi, bankOffsetA, bankOffsetB);
    prescanBlockF4(cacheF4_2, out, blockIndex);
    // No need to store partial min/maxes
}

// ----------------- Alive + Grid  -----------------

__global__ void prescanBool(
    const bool* input, int* out, int* sums,
    const int size, const int blockIndex, const int baseIndex
)
{
    extern __shared__ int cacheInt[];
    int ai, bi, bankOffsetA, bankOffsetB;
    const int shift = ((baseIndex == 0) ? (2 * blockDim.x * blockIdx.x) : baseIndex);
    loadSharedBool(cacheInt, input, size, shift, ai, bi, bankOffsetA, bankOffsetB);
    prescanBlockInt(cacheInt, sums, blockIndex);
    storeSharedInt(out, cacheInt, size, shift, ai, bi, bankOffsetA, bankOffsetB);
}

__global__ void prescanInt(
    const int* input, int* out, int* sums,
    const int size, const int blockIndex, const int baseIndex
)
{
    extern __shared__ int cacheInt_2[];
    int ai, bi, bankOffsetA, bankOffsetB;
    const int shift = ((baseIndex == 0) ? (2 * blockDim.x * blockIdx.x) : baseIndex);
    loadSharedInt(cacheInt_2, input, size, shift, ai, bi, bankOffsetA, bankOffsetB);
    prescanBlockInt(cacheInt_2, sums, blockIndex);
    storeSharedInt(out, cacheInt_2, size, shift, ai, bi, bankOffsetA, bankOffsetB);
}

