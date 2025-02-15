#pragma once
#include "../../headers/clustering/gpu/host.cuh"
#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <random>
#include <algorithm>
#include <utility>



bool testAlivePrefixGPU(const int size, const float aliveFactor, const int maxThreads = 32, const int seed = 42);
bool testBordersPrefixGPU(const int size, const float aliveFactor, const int maxThreads = 32, const int seed = 42, const float maxX = 5000.0, const float maxY = 5000.0);
// bool testGridPrefixGPU(const int size, const float aliveFactor, const int maxThreads = 32, const int seed = 42);
