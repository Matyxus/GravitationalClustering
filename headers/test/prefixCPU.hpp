#pragma once
#include <random>
#include <vector>
#include <iostream>
#include <omp.h>
#include <assert.h>
#include "../utils.hpp"

#define SEED 42

class prefixCPU {
public:
	prefixCPU(int size) : size(size) {
		std::cout << "Allocating random array of size: " << size << std::endl;
		const auto start = std::chrono::steady_clock::now();
		std::default_random_engine bool_generator(SEED);
		// With p = 0.5 you get equal probability for true and false
		std::bernoulli_distribution distribution(0.5);
		// allocate arrays
		indexes = (uint32_t*)malloc(size * sizeof(uint32_t));
		alive = (uint32_t*) malloc(size * sizeof(uint32_t));
		aliveConst = (bool*)malloc(size * sizeof(bool));
		if (indexes == nullptr || alive == nullptr || aliveConst == nullptr) {
			std::cout << "Error: malloc failed!" << std::endl;
			return;
		}
		// Fill arrays
		for (uint32_t i = 0; i < size; i++) {
			indexes[i] = i;
			alive[i] = distribution(bool_generator);
			aliveConst[i] = (bool) alive[i];
			totalAlive += alive[i];
		}
		/*
		for (size_t i = 0; i < 20; i++) {
			std::cout << "Index, alive = (" << indexes[i] << "," << alive[i] << ")" << std::endl;
		}
		*/
		std::cout << "Total alive: " << totalAlive << std::endl;
		std::cout << "Finished generating values, elapsed=" << since(start).count() << "[ms]" << std::endl;
	}
	~prefixCPU() {
		if (indexes != nullptr) {
			free(indexes);
			indexes = nullptr;
		}
		if (alive != nullptr) {
			free(alive);
			alive = nullptr;
		}
		if (aliveConst != nullptr) {
			free(aliveConst);
			aliveConst = nullptr;
		}
	}
	void serial_sort();
	void parallel_sort();
	void parallel_sort2();

private:
	void checkResult(uint32_t* result);
	const int size;
	uint32_t totalAlive = 0;
	uint32_t* indexes = nullptr;
	uint32_t* alive = nullptr;
	bool* aliveConst = nullptr;

};










