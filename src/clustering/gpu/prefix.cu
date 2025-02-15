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
    // this way we can run min/max on both coordinates
    assert(ai + shift < size);
    if (alive[ai + shift]) {
        const float2 first = input[ai + shift];
        cache[ai + bankOffsetA] = float4{ first.x, first.y, first.x, first.y };
    } else { // Use default value
        cache[ai + bankOffsetA] = INVALID_BORDERS;
    }
    if (bi + shift < size && alive[bi + shift]) {
        const float2 second = input[bi + shift];
        cache[bi + bankOffsetB] = float4{ second.x, second.y, second.x, second.y };
    } else { // Duplicate the first loaded value again, cannot use default values
        cache[bi + bankOffsetB] = INVALID_BORDERS;
    }
    /*
    __syncthreads();
    printf("Loaded1 %d at %d-> (%.1f, %.1f, %.1f, %.1f)\n", ai + shift, ai, cache[ai].x, cache[ai].y, cache[ai].z, cache[ai].w);
    printf("Loaded2 %d at %d -> (%.1f, %.1f, %.1f, %.1f)\n", bi + shift, bi, cache[bi].x, cache[bi].y, cache[bi].z, cache[bi].w);
    __syncthreads();
    */
}


template<bool isNP2> __device__ void loadSharedF4(
    float4* cache, const float4* input, const int size, const int shift,
    int& ai, int& bi, int& bankOffsetA, int& bankOffsetB
)
{
    ai = threadIdx.x;
    bi = threadIdx.x + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    assert(ai + shift < size);
    cache[ai + bankOffsetA] = input[ai + shift];
    if (isNP2) { // Compile time decision
        cache[bi + bankOffsetB] = ((bi + shift < size) ? input[bi + shift] : INVALID_BORDERS);
    } else {
        assert(bi + shift < size);
        cache[bi + bankOffsetB] = input[bi + shift];
    }
}
template __device__ void loadSharedF4<true>(float4* cache, const float4* input, const int size, const int shift, int& ai, int& bi, int& bankOffsetA, int& bankOffsetB);
template __device__ void loadSharedF4<false>(float4* cache, const float4* input, const int size, const int shift, int& ai, int& bi, int& bankOffsetA, int& bankOffsetB);

// -------- Alive & Grid --------

template<bool isNP2> __device__ void loadSharedInt(
    int* cache, const int* input, const int size, const int shift,
    int& ai, int& bi, int& bankOffsetA, int& bankOffsetB
)
{
    ai = threadIdx.x;
    bi = threadIdx.x + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    assert(ai + shift < size);
    cache[ai + bankOffsetA] = input[ai + shift];
    if (isNP2) { // Compile time decision
        cache[bi + bankOffsetB] = ((bi + shift < size) ? input[bi + shift] : 0);
    } else {
        assert(bi + shift < size);
        cache[bi + bankOffsetB] = input[bi + shift];
    }
}
template __device__ void loadSharedInt<true>(int* cache, const int* input, const int size, const int shift, int& ai, int& bi, int& bankOffsetA, int& bankOffsetB);
template __device__ void loadSharedInt<false>(int* cache, const int* input, const int size, const int shift, int& ai, int& bi, int& bankOffsetA, int& bankOffsetB);

template<bool isNP2> __device__ void loadSharedBool(
    int* cache, const bool* input, const int size, const int shift,
    int& ai, int& bi, int& bankOffsetA, int& bankOffsetB
)
{
    ai = threadIdx.x;
    bi = threadIdx.x + blockDim.x;
    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    assert(ai + shift < size);
    cache[ai + bankOffsetA] = static_cast<int>(input[ai + shift]);
    if (isNP2) { // Compile time decision
        cache[bi + bankOffsetB] = ((bi + shift < size) ? static_cast<int>(input[bi + shift]) : 0);
    } else {
        assert(bi + shift < size);
        cache[bi + bankOffsetB] = static_cast<int>(input[bi + shift]);
    }
}
template __device__ void loadSharedBool<true>(int* cache, const bool* input, const int size, const int shift, int& ai, int& bi, int& bankOffsetA, int& bankOffsetB);
template __device__ void loadSharedBool<false>(int* cache, const bool* input, const int size, const int shift, int& ai, int& bi, int& bankOffsetA, int& bankOffsetB);


// -------------------------------- Store Shared Mem --------------------------------

template<bool isNP2> __device__ void storeSharedInt(
    int* out, const int* cache, const int size, const int shift,
    int ai, int bi, int bankOffsetA, int bankOffsetB
)
{
    __syncthreads();
    assert(ai + shift < size);
    out[ai + shift] = cache[ai + bankOffsetA];
    if (isNP2) { // Compile time decision
        if (bi + shift < size) {
            out[bi + shift] = cache[bi + bankOffsetB];
        }
    } else {
        assert(bi + shift < size);
        out[bi + shift] = cache[bi + bankOffsetB];
    }
}
template __device__ void storeSharedInt<true>(int* out, const int* cache, const int size, const int shift, int ai, int bi, int bankOffsetA, int bankOffsetB);
template __device__ void storeSharedInt<false>(int* out, const int* cache, const int size, const int shift, int ai, int bi, int bankOffsetA, int bankOffsetB);

// -------------------------------- Save Sum --------------------------------

template<bool store> __device__ void saveSumInt(int* cache, int* out, const int blockIndex) {
    if (threadIdx.x == 0) {
        int index = (2 * blockDim.x) - 1;
        index += CONFLICT_FREE_OFFSET(index);
        // Store total sum to upper level
        if (store) { // Compile time decision
            assert(out != nullptr);
            out[blockIndex] = cache[index];
        }
        // Zero out last
        cache[index] = 0;
    }
}
template __device__ void saveSumInt<true>(int* cache, int* out, const int blockIndex);
template __device__ void saveSumInt<false>(int* cache, int* out, const int blockIndex);


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

template <bool isNP2> __global__ void addInt(int* out, const int* sums, const int size, const int blockOffset, const int baseIndex) {
    __shared__ int sum;
    if (threadIdx.x == 0) {
        sum = sums[blockIdx.x + blockOffset];
    }
    const unsigned int address = 2 * blockDim.x * blockIdx.x + baseIndex + threadIdx.x;
    __syncthreads();
    assert(address < size);
    out[address] += sum;
    if (isNP2) { // Compile-time decision
        if (address + blockDim.x < size) {
            out[address + blockDim.x] += sum;
        }
    } else {
        assert(address + blockDim.x < size);
        out[address + blockDim.x] += sum;
    }
}
template __global__ void addInt<true>(int* out, const int* sums, const int size, const int blockOffset, const int baseIndex);
template __global__ void addInt<false>(int* out, const int* sums, const int size, const int blockOffset, const int baseIndex);

// -------------------------------- Scan Block --------------------------------

// Borders
__device__ void prescanBlockF4(float4* cache, float4* out, const int blockIndex) {
    upScanF4(cache);
    saveSumF4(cache, out, ((blockIndex == 0) ? blockIdx.x : blockIndex));
    // No down sweep
}


// Grid + Alive
template<bool store> __device__ void prescanBlockInt(int* cache, int* out, const int blockIndex) {
    const unsigned int offset = upScanInt(cache);
    saveSumInt<store>(cache, out, ((blockIndex == 0) ? blockIdx.x : blockIndex));
    downScanInt(cache, offset);
}
template __device__ void prescanBlockInt<true>(int* cache, int* out, const int blockIndex);
template __device__ void prescanBlockInt<false>(int* cache, int* out, const int blockIndex);

// -------------------------------- Scan Kernels --------------------------------

// ----------------- Borders -----------------

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

template<bool isNP2>
__global__ void prescanF4(
    const float4* input, float4* out, const int size,
    const int blockIndex, const int baseIndex
) {
    extern __shared__ float4 cacheF4_2[];
    int ai, bi, bankOffsetA, bankOffsetB;
    const int shift = ((baseIndex == 0) ? (2 * blockDim.x * blockIdx.x) : baseIndex);
    loadSharedF4<isNP2>(cacheF4_2, input, size, shift, ai, bi, bankOffsetA, bankOffsetB);
    prescanBlockF4(cacheF4_2, out, blockIndex);
    // No need to store partial min/maxes
}
template __global__ void prescanF4<true>(const float4* input, float4* out, const int size, const int blockIndex, const int baseIndex);
template __global__ void prescanF4<false>(const float4* input, float4* out, const int size, const int blockIndex, const int baseIndex);

// ----------------- Alive + Grid -----------------

template<bool store, bool isNP2> __global__ void prescanBool(
    const bool* input, int* out, int* sums,
    const int size, const int blockIndex, const int baseIndex
)
{
    extern __shared__ int cacheInt[];
    int ai, bi, bankOffsetA, bankOffsetB;
    const int shift = ((baseIndex == 0) ? (2 * blockDim.x * blockIdx.x) : baseIndex);
    loadSharedBool<isNP2>(cacheInt, input, size, shift, ai, bi, bankOffsetA, bankOffsetB);
    prescanBlockInt<store>(cacheInt, sums, blockIndex);
    storeSharedInt<isNP2>(out, cacheInt, size, shift, ai, bi, bankOffsetA, bankOffsetB);
}
template __global__ void prescanBool<true, true>(const bool* input, int* out, int* sums, const int size, const int blockIndex, const int baseIndex);
template __global__ void prescanBool<false, false>(const bool* input, int* out, int* sums, const int size, const int blockIndex, const int baseIndex);
template __global__ void prescanBool<true, false>(const bool* input, int* out, int* sums, const int size, const int blockIndex, const int baseIndex);
template __global__ void prescanBool<false, true>(const bool* input, int* out, int* sums, const int size, const int blockIndex, const int baseIndex);

template<bool store, bool isNP2> __global__ void prescanInt(
    const int* input, int* out, int* sums,
    const int size, const int blockIndex, const int baseIndex
)
{
    extern __shared__ int cacheInt_2[];
    int ai, bi, bankOffsetA, bankOffsetB;
    const int shift = ((baseIndex == 0) ? (2 * blockDim.x * blockIdx.x) : baseIndex);
    loadSharedInt<isNP2>(cacheInt_2, input, size, shift, ai, bi, bankOffsetA, bankOffsetB);
    prescanBlockInt<store>(cacheInt_2, sums, blockIndex);
    storeSharedInt<isNP2>(out, cacheInt_2, size, shift, ai, bi, bankOffsetA, bankOffsetB);
}
template __global__ void prescanInt<true, true>( const int* input, int* out, int* sums, const int size, const int blockIndex, const int baseIndex);
template __global__ void prescanInt<false, false>(const int* input, int* out, int* sums, const int size, const int blockIndex, const int baseIndex);
template __global__ void prescanInt<true, false>(const int* input, int* out, int* sums, const int size, const int blockIndex, const int baseIndex);
template __global__ void prescanInt<false, true>(const int* input, int* out, int* sums, const int size, const int blockIndex, const int baseIndex);

