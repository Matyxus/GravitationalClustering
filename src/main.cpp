#include "../headers/clustering/cpu/ClusteringSerialCPU.hpp"
#include "../headers/clustering/cpu/ClusteringParallelCPU.hpp"
#include "../headers/clustering/gpu/ClusteringGPU.hpp"
#include "../headers/gui/Renderer.hpp"
// #include "../headers/test/prefixCPU.hpp"

void compareState(State& stateA, State& stateB);
void compareCPU(Config& config, Network& network);

int main(int argc, char* argv[]) {
    Config config = Config("./Data/config/clustering_config.json");

    Network network = Network(
        config.getClusteringOptions()->networkPath, 
        config.getClusteringOptions()->dataPath,
        config.getClusteringOptions()->startTime, 
        config.getClusteringOptions()->endTime
    );
   
    // RandomOptions* rng = config.getRandomOptions();
    // assert(rng != nullptr);
    // ClusteringGPU clustering = ClusteringGPU(&config, rng);
    
    // ClusteringGPU clustering = ClusteringGPU(&config, &network);
    /*
    ClusteringSerialCPU clustering = ClusteringSerialCPU(&config, &network);
    for (size_t i = 0; i < 25; i++) {
        clustering.step();
    }
    */
    ClusteringSerialCPU clustering = ClusteringSerialCPU(&config, &network);
    
    Renderer renderer = Renderer(clustering.getGrid().borders);
    State &state = clustering.getState();
    int prevState = PAUSE;
    int eventState = PAUSE;
    std::cout << "Current event: " << eventNames[eventState] << std::endl;
    while (renderer.isRunning()) {
        prevState = eventState;
        eventState = renderer.pollEvents(eventState);
        if (eventState != prevState) {
            std::cout << "Current event: " << eventNames[eventState] << std::endl;
        }
        switch (eventState) {
            case QUIT:
                break;
            case NEXT:
                std::cout << "Next is not fully implemented yet!" << std::endl;
                clustering.step();
                eventState = PAUSE;
                break;
            case PREVIOUS:
                std::cout << "Previous is not implemented yet!" << std::endl;
                eventState = PAUSE;
                break;
            case PAUSE:
                break; // Pass
            case RUN:
                clustering.step();
                break;
            default:
                break;
        }
        if (!renderer.plotPlanets(state, config.getClusteringOptions()->multiplier)) {
            break;
        }
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
	return 0;
}


void compareState(State& stateA, State& stateB) {
    assert(stateA.isInitialized() && stateB.isInitialized());
    assert(stateA.iteration == stateB.iteration);
    assert(stateA.numEdges == stateB.numEdges);
    assert(stateA.numAlive == stateB.numAlive);
    // Compare arrays
    for (int i = 0; i < stateA.numEdges; i++) {
        assert(stateA.positions[i].x == stateB.positions[i].x && stateA.positions[i].y == stateB.positions[i].y);
        assert(stateA.movements[i].x == stateB.movements[i].x && stateA.movements[i].y == stateB.movements[i].y);
        assert(stateA.weigths[i] == stateB.weigths[i]);
        assert(stateA.clusters[i].x == stateB.clusters[i].x && stateA.clusters[i].y == stateB.clusters[i].y);
        assert(stateA.indexes[i] == stateB.indexes[i]);
        assert(stateA.merges[i] == stateB.merges[i]);
        assert(stateA.alive[i] == stateB.alive[i]);
    }
}

void compareCPU(Config &config, Network &network) {
    std::cout << "Comparing serial vs parallel" << std::endl;
    ClusteringParallelCPU parallel = ClusteringParallelCPU(&config, &network);
    ClusteringSerialCPU serial = ClusteringSerialCPU(&config, &network);
    State& stateA = parallel.getState();
    State& stateB = serial.getState();
    compareState(stateA, stateB);
    for (size_t i = 0; i < 140; i++) {
        serial.step();
        parallel.step();
        compareState(stateA, stateB);
    }
}
