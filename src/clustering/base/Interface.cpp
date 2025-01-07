#include "../../../headers/clustering/base/Interface.hpp"


Interface::Interface(Config* config, Network* network) {
	std::cout << "Initializing Clustering Interface for Netowork" << std::endl;
	if (config == nullptr || network == nullptr) {
		std::cout << "Error: Configuration or Network are invalid!" << std::endl;
		return;
	}
	plotOptions = config->getPlotOptions();
	clusteringOptions = config->getClusteringOptions();
	this->network = network;
	assert(network->getEdges().size() < std::numeric_limits<int>::max());
	state.numEdges = static_cast<int>(network->getEdges().size());
	state.numAlive = state.numEdges;
}


Interface::Interface(Config* config, RandomOptions* randomOptions) {
	std::cout << "Initializing Clustering Interface for RNG" << std::endl;
	if (config == nullptr || randomOptions == nullptr) {
		std::cout << "Error: Configuration is invalid!" << std::endl;
		return;
	}
	plotOptions = config->getPlotOptions();
	clusteringOptions = config->getClusteringOptions();
	state.numEdges = randomOptions->size;
	state.numAlive = randomOptions->size; // Needs to be decreased when generating points
	this->randomOptions = randomOptions;
}

bool Interface::checkGrid() {
	uint16_t visited, index;
	for (size_t i = 0; i < grid.size; i++) {
		visited = 0;
		index = grid.gridMap[i];
		while (index < state.numEdges && grid.cells[grid.sortedCells[index]] == i) {
			visited++;
			index++;
		}
		assert(visited == grid.binCounts[i]);
	}
	return true;
}


bool Interface::generateRandomValues() {
	std::cout << "Generating random values" << std::endl;
	if (randomOptions == nullptr) {
		std::cout << "Error: RandomOptions are not initialized !" << std::endl;
		return false;
	}
	// ----------------- Initialize State -----------------
	state.positions = (float2*)malloc(state.numEdges * sizeof(float2));
	state.movements = (float2*)malloc(state.numEdges * sizeof(float2));
	state.weigths = (float*)malloc(state.numEdges * sizeof(float));
	state.clusters = (int2*)malloc(state.numEdges * sizeof(int2));
	state.indexes = (int*)malloc(state.numEdges * sizeof(int));
	state.merges = (int*)malloc(state.numEdges * sizeof(int));
	state.alive = (bool*)malloc(state.numEdges * sizeof(bool));
	assert(state.isInitialized());
	// ----------------- Initialize RNG -----------------
	std::mt19937 floatGen(randomOptions->seed);
	std::default_random_engine boolGen(randomOptions->seed);
	std::uniform_real_distribution<float> distrX(0, randomOptions->max_y); // define the range for X
	std::uniform_real_distribution<float> distrY(0, randomOptions->max_y); // define the range for Y
	std::uniform_real_distribution<float> distrCI(0.f, 1.f);
	std::bernoulli_distribution distrAlive(randomOptions->aliveFactor);
	// Assign random values to arrays 
	for (int i = 0; i < state.numEdges; i++) {
		state.alive[i] = distrAlive(boolGen);
		state.numAlive -= (!state.alive[i]);
		state.positions[i] = { distrX(floatGen), distrY(floatGen) };
		state.movements[i] = { 0.f, 0.f };
		state.weigths[i] = (distrCI(floatGen) + clusteringOptions->offset) * clusteringOptions->multiplier;
		state.indexes[i] = i;
		state.clusters[i] = int2{ i, -1 };
		state.merges[i] = -1;
	}
	std::cout << "Successfully generated random values" << std::endl;
	return true;
}
