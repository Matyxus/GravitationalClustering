#include "../../headers/test/prefixGPU.cuh"
#include <stdio.h>
#include <random>
#include <algorithm>
#include <utility>


// ------- Forwards ------- 

void prescanArrayRecursiveInt(int* outArray, const int* inArray, int numElements, int level, const int maxThreads);
void prescanArrayF4(const float2* inArray, const bool* alive, int numElements, const int maxThreads);
bool checkArrayInt(int* gpuArray, int* cpuArray, const int size);
bool checkArrayF4(float4 resultGPU, float2* cpuArray, const bool* alive, const int size);

// ------------------------------------------- Tests ------------------------------------------- 


bool testBordersPrefixGPU(const int size, const float aliveFactor, const int maxThreads, const int seed, const float maxX, const float maxY) {
    printf("Testing borders prefix scan, (size, aliveFactor, maxThreads, seed) = (%d, %f, %d, %d)\n", size, aliveFactor, maxThreads, seed);
    // Alocate alive arrays
    bool *aliveCPU = (bool*)malloc(size * sizeof(bool));
    bool *aliveGPU = nullptr;
    CHECK_ERROR(cudaMalloc((void**)&aliveGPU, size * sizeof(int)));
    assert(aliveCPU != nullptr);
    // Alocate border arrays
    float2 *bordersCPU = (float2*)malloc(size * sizeof(float2));
    float2 *bordersGPU = nullptr;
    CHECK_ERROR(cudaMalloc((void**)&bordersGPU, size * sizeof(float2)));
    assert(bordersCPU != nullptr);
    allocateStateHelpers(size, maxThreads);
    // ----------------- Initialize RNG -----------------
    std::mt19937 floatGen(seed);
    std::uniform_real_distribution<float> distrX(0.f, maxX); // define the range for X
    std::uniform_real_distribution<float> distrY(0.f, maxY); // define the range for Y
    std::default_random_engine boolGen(seed);
    std::bernoulli_distribution distrAlive(aliveFactor);
    for (int i = 0; i < size; i++) {
        aliveCPU[i] = distrAlive(boolGen);
        bordersCPU[i] = {distrX(floatGen), distrY(floatGen)};
    }
    aliveCPU[0] = true; // Make 0th element alive
    CHECK_ERROR(cudaMemcpy(aliveGPU, aliveCPU, size * sizeof(bool), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(bordersGPU, bordersCPU, size * sizeof(float2), cudaMemcpyHostToDevice));
    std::cout << "Finished allocating arrays of size: " << size << std::endl;
    prescanArrayF4(bordersGPU, aliveGPU, size, maxThreads);
    CHECK_ERROR(cudaDeviceSynchronize());
    // Get the final result and copy it to CPU
    float4 borders;
    CHECK_ERROR(cudaMemcpy(&borders, bordersBlocks->blocks[bordersBlocks->levels - 1], sizeof(float4), cudaMemcpyDeviceToHost));
    bool correct = checkArrayF4(borders, bordersCPU, aliveCPU, size);
    std::cout << "Correct: " << correct << std::endl;
    // ------ Free ------
    CHECK_ERROR(cudaFree(aliveGPU));
    CHECK_ERROR(cudaFree(bordersGPU));
    freeHelpers();
    free(aliveCPU);
    free(bordersCPU);
    return correct;
}


bool testAlivePrefixGPU(const int size, const float aliveFactor, const int maxThreads, const int seed) {
    printf("Testing alive prefix scan, (size, aliveFactor, maxThreads, seed) = (%d, %f, %d, %d)\n", size, aliveFactor, maxThreads, seed);
    int* list = (int*)malloc(size * sizeof(int));
    assert(list != nullptr);
    int* inArray = nullptr;
    // ----------------- Initialize RNG -----------------
    std::mt19937 floatGen(seed);
    std::default_random_engine boolGen(seed);
    std::bernoulli_distribution distrAlive(aliveFactor);
    list[0] = 1;
    for (int i = 1; i < size; i++) {
        list[i] = static_cast<int>(distrAlive(boolGen));
    }
    // ----------------- Allocate cuda -----------------
    allocateStateHelpers(size, maxThreads);
    CHECK_ERROR(cudaMalloc((void**)&inArray, size * sizeof(int)));
    CHECK_ERROR(cudaMemcpy(inArray, list, size * sizeof(int), cudaMemcpyHostToDevice));
    std::cout << "Finished allocating array of size: " << size << std::endl;
    // ----------------- run prefix scan -----------------
    assert(aliveBlocks != nullptr);
    prescanArrayRecursiveInt(aliveBlocks->helper, inArray, size, 0, maxThreads);
    CHECK_ERROR(cudaDeviceSynchronize());
    bool correct = checkArrayInt(aliveBlocks->helper, list, size);
    std::cout << "Correct: " << correct << std::endl;
    // ------ Free ------
    CHECK_ERROR(cudaFree(inArray));
    freeHelpers();
    free(list);
    return correct;
}


bool testGridPrefixGPU(const int size, const float aliveFactor, const int maxThreads = 32, const int seed = 42) {
    /*
    int* list = (int*)malloc(size * sizeof(int));
    assert(list != nullptr);
    int* inArray, * outArray;
    // const float max_x = 5000, max_y = 5000;
    // ----------------- Initialize RNG -----------------
    std::mt19937 floatGen(seed);
    std::default_random_engine boolGen(seed);
    std::bernoulli_distribution distrAlive(aliveFactor);
    list[0] = 1;
    for (int i = 1; i < size; i++) {
        list[i] = static_cast<int>(distrAlive(boolGen));
    }
    // ----------------- Allocate cuda -----------------
    allocateStateHelpers(size, maxThreads);
    CHECK_ERROR(cudaMemcpy(inArray, list, size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemset(outArray, 0, size * sizeof(int)));
    std::cout << "Finished allocating array of size: " << size << std::endl;
    // ----------------- run prefix scan -----------------
    prescanArrayRecursiveInt(outArray, inArray, size, 0);
    CHECK_ERROR(cudaDeviceSynchronize());
    bool correct = checkArray(outArray, list, size);
    std::cout << "Correct: " << correct << std::endl;
    // ------ Free ------
    CHECK_ERROR(cudaFree(inArray));
    CHECK_ERROR(cudaFree(outArray));
    freeHelpers();
    free(list);
    */
    return true;
}


// ------------------------------------------- Recursion parallel reduction ------------------------------------------- 


void prescanArrayRecursiveInt(int* outArray, const int* inArray, int numElements, int level, const int maxThreads) {
    Level lvl = Level(numElements, maxThreads, sizeof(int));
    // setup execution parameters if NP2, we process the last block separately
    std::cout << " -------------------------------- " << std::endl;
    std::cout << "Prescan level: " << level << ", blocks: " << lvl.blocks << ", np2: " << lvl.lastBlock << std::endl;
    std::cout << "Threads: " << lvl.threads << ", size: " << numElements << std::endl;
    if (lvl.lastBlock) {
        std::cout << "lastBlockSize: " << lvl.lastElementsCount << ", threadsLastBlock: " << lvl.lastThreads << std::endl;
    }
    if (lvl.blocks > 1) {
        prescanInt<true, false><<<lvl.grid, lvl.threads, lvl.sharedMem>>>(inArray, outArray, aliveBlocks->blocks[level], numElements - lvl.lastBlock * lvl.lastElementsCount, 0, 0);
        if (lvl.lastBlock) {
            prescanInt<true, true><<<1, lvl.lastThreads, lvl.lastSharedMem>>>(inArray, outArray, aliveBlocks->blocks[level], numElements, lvl.blocks - 1, numElements - lvl.lastElementsCount);
        }
        prescanArrayRecursiveInt(aliveBlocks->blocks[level], aliveBlocks->blocks[level], lvl.blocks, level + 1, maxThreads);
        addInt<false><<<lvl.grid, lvl.threads>>>(outArray, aliveBlocks->blocks[level], numElements - lvl.lastBlock * lvl.lastElementsCount, 0, 0);
        if (lvl.lastBlock) {
            addInt<true><<<1, lvl.lastThreads>>>(outArray, aliveBlocks->blocks[level], numElements, lvl.blocks - 1, numElements - lvl.lastElementsCount);
        }
    } else if (isPowerOfTwo(numElements)) {
        prescanInt<false, false><<<lvl.grid, lvl.threads, lvl.sharedMem>>>(inArray, outArray, nullptr, numElements, 0, 0);
    } else {
        prescanInt<false, true><<<lvl.grid, lvl.threads, lvl.sharedMem>>>(inArray, outArray, nullptr, numElements, 0, 0);
    }
}

void prescanArrayF4(const float2* inArray, const bool* alive, int numElements, const int maxThreads) {
    // Compute number of blocks & threads needed
    Level lvl = Level(numElements, maxThreads, sizeof(float4));
    int i = 0;
    std::cout << " -------------------------------- " << std::endl;
    std::cout << "Prescan level: 0"  << ", blocks: " << lvl.blocks << ", np2: " << lvl.lastBlock << std::endl;
    std::cout << "Threads: " << lvl.threads << ", size: " << numElements << std::endl;
    if (lvl.lastBlock) {
        std::cout << "lastBlockSize: " << lvl.lastElementsCount << ", threadsLastBlock: " << lvl.lastThreads << std::endl;
    }
    // Peform parallel reduction UP
    prescanF2<<<lvl.grid, lvl.threads, lvl.sharedMem>>>(inArray, bordersBlocks->blocks[0], alive, numElements - lvl.lastBlock * lvl.lastElementsCount, 0, 0);
    if (lvl.lastBlock) {
        prescanF2<<<1, lvl.lastThreads, lvl.lastSharedMem>>>(inArray, bordersBlocks->blocks[0], alive, numElements, lvl.blocks - 1, numElements - lvl.lastElementsCount);
    }
    // CHECK_ERROR(cudaDeviceSynchronize());
    // exit(1);
    for (i = 1; i < bordersBlocks->levels - 1; i++) {
        numElements = lvl.blocks;
        lvl = Level(numElements, maxThreads, sizeof(float4));
        std::cout << " -------------------------------- " << std::endl;
        std::cout << "Prescan level: " << i << ", blocks: " << lvl.blocks << ", np2: " << lvl.lastBlock << std::endl;
        std::cout << "Threads: " << lvl.threads << ", size: " << numElements << std::endl;
        if (lvl.lastBlock) {
            std::cout << "lastBlockSize: " << lvl.lastElementsCount << ", threadsLastBlock: " << lvl.lastThreads << std::endl;
        }
        prescanF4<false><<<lvl.grid, lvl.threads, lvl.sharedMem>>>(bordersBlocks->blocks[i - 1], bordersBlocks->blocks[i], numElements, 0, 0);
        if (lvl.lastBlock) {
            prescanF4<true><<<1, lvl.lastThreads, lvl.lastSharedMem>>>(bordersBlocks->blocks[i - 1], bordersBlocks->blocks[i], numElements, lvl.blocks - 1, numElements - lvl.lastElementsCount);
        }
    }
    numElements = lvl.blocks;
    lvl = Level(numElements, maxThreads, sizeof(float4));
    std::cout << " -------------------------------- " << std::endl;
    std::cout << "Prescan level: " << i << ", blocks: " << lvl.blocks << ", np2: " << lvl.lastBlock << std::endl;
    std::cout << "Threads: " << lvl.threads << ", size: " << numElements << std::endl;
    if (isPowerOfTwo(numElements)) {
        prescanF4<false><<<lvl.grid, lvl.threads, lvl.sharedMem>>>(bordersBlocks->blocks[i - 1], bordersBlocks->blocks[i], numElements, 0, 0);
    } else {
        prescanF4<true><<<lvl.grid, lvl.threads, lvl.sharedMem>>>(bordersBlocks->blocks[i - 1], bordersBlocks->blocks[i], numElements, 0, 0);
    }
}


// ------------------------------------------- Correctness checks ------------------------------------------- 

bool checkArrayInt(int* gpuArray, int* cpuArray, const int size) {
    int* cpuCopy = (int*)malloc(size * sizeof(int));
    int* cpuPrefix = (int*)malloc(size * sizeof(int));
    assert(cpuCopy != nullptr && cpuPrefix != nullptr);
    CHECK_ERROR(cudaMemcpy(cpuCopy, gpuArray, size * sizeof(int), cudaMemcpyDeviceToHost));
    cpuPrefix[0] = 0;
    for (int i = 0; i < size - 1; i++) {
        cpuPrefix[i + 1] = (cpuArray[i] + cpuPrefix[i]);
    }
    bool correct = true;
    for (int i = 0; i < size; i++) {
        correct = (cpuCopy[i] == cpuPrefix[i]);
        if (!correct) {
            std::cout << "Arr[" << i << "]=" << cpuCopy[i] << " | " << cpuPrefix[i] << std::endl;
            break;
        }
    }
    free(cpuCopy);
    free(cpuPrefix);
    return correct;
}


bool checkArrayF4(float4 resultGPU, float2* cpuArray, const bool* alive, const int size) {
    float4 resultCPU = INVALID_BORDERS;
    // Compute the smallest coordinates
    for (int i = 0; i < size; i++) {
        // Skip dead point
        if (!alive[i]) {
            continue;
        }
        resultCPU.x = std::max(resultCPU.x, cpuArray[i].x);
        resultCPU.y = std::max(resultCPU.y, cpuArray[i].y);
        resultCPU.z = std::min(resultCPU.z, cpuArray[i].x);
        resultCPU.w = std::min(resultCPU.w, cpuArray[i].y);
    }
    std::cout << "CPU result: ((" << resultCPU.z << "," << resultCPU.w << "), (" << resultCPU.x << "," << resultCPU.y << "))" << std::endl;
    std::cout << "GPU result: ((" << resultGPU.z << "," << resultGPU.w << "), (" << resultGPU.x << "," << resultGPU.y << "))" << std::endl;
    return (resultCPU.x == resultGPU.x && resultCPU.y == resultGPU.y && resultCPU.z == resultGPU.z && resultCPU.w == resultGPU.w);
}
