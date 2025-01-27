#include "../headers/clustering/cpu/ClusteringSerialCPU.hpp"
#include "../headers/clustering/cpu/ClusteringParallelCPU.hpp"
#include "../headers/clustering/gpu/ClusteringGPU.hpp"
#include "../headers/gui/Renderer.hpp"
// #include "../headers/test/prefixCPU.hpp"

void compareState(State& stateA, State& stateB);
void compareGrid(Grid& gridA, Grid& gridB, const int numClusters);
void compareCPU(Config& config, Network& network);

int main(int argc, char* argv[]) {
    Config config = Config("./Data/config/clustering_config.json");
    auto data = config.getDataOptions();
    Network network = Network(
        data->networkOptions.path,
        data->networkOptions.edgeData,
        data->networkOptions.startTime,
        data->networkOptions.endTime,
        data->networkOptions.offset
    );
    ClusteringGPU clustering = ClusteringGPU(&config, &network);



    // clustering.step();
    /*
    Renderer renderer = Renderer(clustering.getGrid().borders, config.getGuiOptions()->windowOptions);
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
        if (!renderer.plotPlanets(state, 10.f)) {
            break;
        }
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    */
	return 0;
}


void compareState(State& stateA, State& stateB) {
    assert(stateA.isInitialized() && stateB.isInitialized());
    assert(stateA.iteration == stateB.iteration);
    assert(stateA.size == stateB.size);
    assert(stateA.numAlive == stateB.numAlive);
    // Compare arrays
    for (int i = 0; i < stateA.size; i++) {
        assert(stateA.positions[i].x == stateB.positions[i].x && stateA.positions[i].y == stateB.positions[i].y);
        assert(stateA.movements[i].x == stateB.movements[i].x && stateA.movements[i].y == stateB.movements[i].y);
        assert(stateA.weigths[i] == stateB.weigths[i]);
        assert(stateA.clusters[i].x == stateB.clusters[i].x && stateA.clusters[i].y == stateB.clusters[i].y);
        assert(stateA.indexes[i] == stateB.indexes[i]);
        assert(stateA.merges[i] == stateB.merges[i]);
        assert(stateA.alive[i] == stateB.alive[i]);
    }
}

void compareGrid(Grid& gridA, Grid& gridB, const int numClusters) {
    assert(gridA.isInitialized() && gridB.isInitialized());
    assert(gridA.borders.x == gridB.borders.x && gridA.borders.y == gridB.borders.y);
    assert(gridA.borders.z == gridB.borders.z && gridA.borders.w == gridB.borders.w);
    assert(gridA.size == gridB.size && gridA.rows == gridB.rows && gridA.cols == gridB.cols);
    // Compare arrays
    for (int i = 0; i < numClusters; i++) {
        assert(gridA.cells[i] == gridB.cells[i]);
    }
    for (int i = 0; i < gridA.size; i++) {
        assert(gridA.binCounts[i] == gridB.binCounts[i]);
    }

}

void compareCPU(Config &config, Network &network) {
    std::cout << "Comparing serial vs parallel" << std::endl;
    ClusteringParallelCPU parallel = ClusteringParallelCPU(&config, &network);
    ClusteringSerialCPU serial = ClusteringSerialCPU(&config, &network);
    State& stateA = parallel.getState();
    State& stateB = serial.getState();
    Grid& gridA = parallel.getGrid(); 
    Grid& gridB = serial.getGrid();
    compareState(stateA, stateB);
    for (size_t i = 0; i < 140; i++) {
        serial.step();
        parallel.step();
        compareState(stateA, stateB);
        compareGrid(gridA, gridB, stateA.size);
    }
}
