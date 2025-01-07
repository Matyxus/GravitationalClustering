#include "../../headers/test/prefixCPU.hpp"

// ~628ms for 10M size
void prefixCPU::serial_sort() {
	std::cout << "Running serial sort" << std::endl;
	const auto start = std::chrono::steady_clock::now();
	uint32_t* result = (uint32_t*)calloc(totalAlive, sizeof(uint32_t));
	assert(result != nullptr);
	uint32_t j = 0;
	for (uint32_t i = 0; i < size; i++) {
		if (alive[i]) {
			result[j] = indexes[i];
			j++;
		}
	}
	std::cout << "Finished serial sort, elapsed=" << since(start).count() << "[ms]" << std::endl;
	checkResult(result);
	free(result);
	return;
}


// ~150ms for 10M size
void prefixCPU::parallel_sort() {
	std::cout << "Running parallel sort" << std::endl;
	const auto timeStart = std::chrono::steady_clock::now();
	const int threads = omp_get_max_threads();
	const int work = size / threads;
	uint32_t* result = (uint32_t*)calloc(totalAlive, sizeof(uint32_t));
	uint32_t* partial = (uint32_t*)calloc(threads, sizeof(uint32_t));
	std::cout << "Max num threads: " << threads << ", work per thread: " << work << std::endl;
	assert(partial != nullptr && result != nullptr);
	#pragma omp parallel 
	{
		// Compute partial sums of the array for each thread
		const int tid = omp_get_thread_num();
		const int a = (tid * work);
		const int b = (tid == threads - 1) ? size : (a + work);
		for (int i = a + 1; i < b; i++) {
			alive[i] += alive[i - 1];
		}
		partial[tid] = alive[b - 1];
		// Sync threads, sum partial sums for each thread
		#pragma omp barrier
		uint32_t offset = 0;
		for (int i = 0; i < tid; i++) {
			offset += partial[i];
		}
		/*
		#pragma omp critical 
		{	
			std::cout << "------------------------------" << std::endl;
			std::cout << "Thread: " << tid << " writing between: (" << a << "," << (b-1) << "), prefix=" << before << std::endl;
			if (alive[a] == 1) {
				std::cout << "Result[" << before << "]=" << indexes[a] << std::endl;
			}
			for (int i = a; i < b-1; i++) {
				std::cout << "Alive[" << i << "]=" << alive[i];
				std::cout << " =?= Alive[" << (i + 1) << "]=" << alive[i + 1] << std::endl;
				// assert(i + 1 < size);
				if (alive[i] != alive[i + 1]) {
					std::cout << "Result[" << (alive[i] + before) << "]=" << indexes[i + 1] << std::endl;
				}
			}
		}
		*/
		// Copy first element (needed if alive[a] == 1)
		if (alive[a] == 1) {
			result[offset] = indexes[a];
		}
		// Sort indexes
		for (int i = a; i < b-1; i++) {
			if (alive[i] != alive[i + 1]) {
				// assert(alive[i] + offset < totalAlive);
				// assert(result[alive[i] + offset] == 0);
				result[alive[i] + offset] = indexes[i+1];
			}
		}
	}
	std::cout << "Finished parallel sort, elapsed=" << since(timeStart).count() << "[ms]" << std::endl;
	checkResult(result);
	free(partial);
	free(result);
	return;
}


// ~150ms for 10M size
void prefixCPU::parallel_sort2() {
	std::cout << "Running parallel sort2" << std::endl;
	const auto timeStart = std::chrono::steady_clock::now();
	const int threads = omp_get_max_threads();
	const int chunk = size / threads;
	uint32_t* result = (uint32_t*)calloc(totalAlive, sizeof(uint32_t));
	uint32_t* partial = (uint32_t*)calloc(threads, sizeof(uint32_t));
	std::cout << "Max num threads: " << threads << ", chunk: " << chunk << std::endl;
	assert(partial != nullptr && result != nullptr);
	uint32_t sum = 0;
	int start = 0;
	int end = chunk;
	for (int i = 0; i < threads - 1; i++) {
		// Compute partial sums of the array for each thread
		#pragma omp parallel reduction(+:sum)
		{
			for (int j = omp_get_thread_num() + start; j < end; j += threads) {
				sum += alive[j];
			}
		}
		start += chunk;
		end += chunk;
		partial[i + 1] = sum;
	}
	// Sort results
	#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		const int a = tid * chunk;
		const int b = (tid == threads - 1) ? size : (a + chunk);
		uint32_t offset = partial[tid];
		// Write to indexes
		for (int i = a; i < b; i++) {
			if (alive[i]) {
				// assert(offset < totalAlive);
				// assert(result[offset] == 0);
				result[offset] = indexes[i];
				offset++;
			}
		}
	}
	std::cout << "Finished parallel sort2, elapsed=" << since(timeStart).count() << "[ms]" << std::endl;
	checkResult(result);
	free(partial);
	free(result);
	return;
}


// ------------------------------------ Utils ------------------------------------

void prefixCPU::checkResult(uint32_t* result) {
	for (size_t i = 0; i < totalAlive; i++) {
		if (!(aliveConst[result[i]])) {
			std::cout << "Error, index: " << result[i] << " is not alive!" << std::endl;
			exit(1);
		}
		aliveConst[result[i]] = false; // Check against duplicates
	}
	// Put values back
	for (size_t i = 0; i < totalAlive; i++) {
		aliveConst[result[i]] = true;
	}
	return;
}

