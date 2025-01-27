#include "../../../headers/clustering/base/Interface.hpp"


Interface::Interface(Config* config, Network* network) {
	std::cout << "Initializing Clustering Interface for Network" << std::endl;
	if (config == nullptr || network == nullptr) {
		std::cout << "Error: Configuration or Network are invalid!" << std::endl;
		return;
	}
	clusteringOptions = config->getClusteringOptions();
	state.allocate(static_cast<int>(network->getEdges().size()));
	assert(state.isInitialized());
	state.numAlive = state.size;
	// Assign values to arrays
	std::pair<float, float> position;
	for (Edge*& edge : network->getEdges()) {
		position = edge->getCentroid();
		state.alive[edge->identifier] = true;
		state.positions[edge->identifier] = { position.first, position.second };
		state.movements[edge->identifier] = { 0.f, 0.f };
		state.weigths[edge->identifier] = edge->getCongestionIndex()  * clusteringOptions->params.multiplier;
		state.indexes[edge->identifier] = edge->identifier;
		state.clusters[edge->identifier] = int2{ edge->identifier, -1 };
		state.merges[edge->identifier] = -1;
	}
	std::cout << "Successfully assigned: '" << state.size << "' edges as clusters" << std::endl;
}


Interface::Interface(Config* config) {
	std::cout << "Initializing Clustering Interface for RNG" << std::endl;
	if (config == nullptr) {
		std::cout << "Error: Configuration is invalid!" << std::endl;
		return;
	} else if (!config->getDataOptions()->use_rng){
		std::cout << "Error: Configuration is not set to use RNG!" << std::endl;
		return;
	}
	clusteringOptions = config->getClusteringOptions();
	RandomOptions randomOptions = config->getDataOptions()->randomOptions;
	state.allocate(randomOptions.size);
	assert(state.isInitialized());
	// ----------------- Initialize RNG -----------------
	std::mt19937 floatGen(randomOptions.seed);
	std::default_random_engine boolGen(randomOptions.seed);
	std::uniform_real_distribution<float> distrX(0.f, randomOptions.max_y); // define the range for X
	std::uniform_real_distribution<float> distrY(0.f, randomOptions.max_y); // define the range for Y
	std::uniform_real_distribution<float> distrCI(0.f + config->getDataOptions()->networkOptions.offset, 1.f);
	std::bernoulli_distribution distrAlive(randomOptions.aliveFactor);
	// Assign random values to arrays 
	for (int i = 0; i < state.size; i++) {
		state.alive[i] = distrAlive(boolGen);
		state.numAlive -= (!state.alive[i]);
		state.positions[i] = { distrX(floatGen), distrY(floatGen) };
		state.movements[i] = { 0.f, 0.f };
		state.weigths[i] = distrCI(floatGen) * clusteringOptions->params.multiplier;
		state.indexes[i] = i;
		state.clusters[i] = int2{ i, -1 };
		state.merges[i] = -1;
	}
	// Force 0th value to be alive (due to merging strategy, this index is permanent, i.e. never merged by other)
	state.numAlive += (!state.alive[0]);
	state.alive[0] = true;
	std::cout << "Successfully generated random: '" << state.size << "' clusters" << std::endl;
}
