#include "../../../headers/clustering/base/Config.hpp"

Config::~Config() {
    std::cout << "Freeing Config ... " << std::endl;
    if (clusteringOptions != nullptr) {
        delete clusteringOptions;
        clusteringOptions = nullptr;
    }
    if (plotOptions != nullptr) {
        delete plotOptions;
        plotOptions = nullptr;
    }
};

bool Config::loadConfig(const std::string path) {
    std::cout << "Loading json config file from: " << path << std::endl;
    // Try to load file
    std::ifstream file(path);
    if (!file.good()) {
        std::cout << "Error: File does not exist!" << std::endl;
        loaded = false;
        return false;
    }
    // Try to parse file
    nlohmann::json data;
    try {
        data = nlohmann::json::parse(file);
    }  catch (const nlohmann::json::parse_error& e) {
        std::cerr << "Error: Failed to parse JSON! " << e.what() << std::endl;
        loaded = false;
        return false;
    }
    // TODO: Check the data against schema
    // Load data
    plotOptions = new PlotOptions{
        data["plotting"]["frequency"].get<uint16_t>(),
        data["plotting"]["clusterSize"].get<uint16_t>(),
        data["plotting"]["heatmap"].get<bool>(),
        data["plotting"]["planets"].get<bool>()
    };
    std::cout << "Loading clustering" << std::endl;
    clusteringOptions = new ClusteringOptions{
        data["clustering"]["networkPath"].get<std::string>(),
        data["clustering"]["dataPath"].get<std::string>(),
        data["clustering"]["iterations"].get<uint16_t>(),
        data["clustering"]["multiplier"].get<float>(),
        data["clustering"]["startTime"].get<float>(),
        data["clustering"]["endTime"].get<float>(),
        data["clustering"]["radius"].get<float>(),
        data["clustering"]["offset"].get<float>(),
        ResizingOptions(
            data["stateResize"]["allowed"].get<bool>(), 
            data["stateResize"]["percentage"].get<float>(),
            data["stateResize"]["minSize"].get<int>(),
            data["stateResize"]["limit"].get<int>(),
            data["stateResize"]["frequency"].get<int>()
        ),
        ResizingOptions(
            data["gridResize"]["allowed"].get<bool>(),
            data["gridResize"]["percentage"].get<float>(),
            data["gridResize"]["minSize"].get<int>(),
            data["gridResize"]["limit"].get<int>(),
            data["gridResize"]["frequency"].get<int>()
        ),
        DeviceOptions(
            data["device"]["mode"].get<std::string>(),
            data["device"]["threads"].get<int>()
        )
    };
    // Load RNG 
    if (data.count("rng") == 1) {
        randomOptions = new RandomOptions(
            data["rng"]["size"].get<int>(),
            data["rng"]["seed"].get<int>(),
            data["rng"]["aliveFactor"].get<float>(),
            data["rng"]["max_x"].get<float>(),
            data["rng"]["max_y"].get<float>()
        );
    }
    std::cout << "-------------- Clustering Params --------------" << std::endl;
    std::cout << "multiplier: " << clusteringOptions->multiplier << std::endl;
    std::cout << "radius: " << clusteringOptions->radius << std::endl;
    std::cout << "offset: " << clusteringOptions->offset << std::endl;
    std::cout << "#State resizing"  << std::endl;
    std::cout << "-resize: " << clusteringOptions->stateResize.allowed << std::endl;
    std::cout << "-percentage: " << clusteringOptions->stateResize.percentage << std::endl;
    std::cout << "-minSize: " << clusteringOptions->stateResize.minSize << std::endl;
    std::cout << "-limit: " << clusteringOptions->stateResize.limit << std::endl;
    std::cout << "-frequency: " << clusteringOptions->stateResize.limit << std::endl;
    std::cout << "#Grid resizing" << std::endl;
    std::cout << "-resize: " << clusteringOptions->gridResize.allowed << std::endl;
    std::cout << "-percentage: " << clusteringOptions->gridResize.percentage << std::endl;
    std::cout << "-minSize: " << clusteringOptions->gridResize.minSize << std::endl;
    std::cout << "-limit: " << clusteringOptions->gridResize.limit << std::endl;
    std::cout << "-frequency: " << clusteringOptions->gridResize.limit << std::endl;
    std::cout << "#Device" << std::endl;
    std::cout << "-mode: " << clusteringOptions->device.mode << std::endl;
    std::cout << "-threads: " << clusteringOptions->device.threads << std::endl;
    if (randomOptions != nullptr) {
        std::cout << "#RNG" << std::endl;
        std::cout << "-size: " << randomOptions->size << std::endl;
        std::cout << "-seed: " << randomOptions->seed << std::endl;
        std::cout << "-aliveFactor: " << randomOptions->aliveFactor << std::endl;
        std::cout << "-max_x: " << randomOptions->max_x << std::endl;
        std::cout << "-max_y: " << randomOptions->max_y << std::endl;
    }
    std::cout << "-------------- ----------------- --------------" << std::endl;
    // Load road network & congestion indexes
    /*
    network = new Network();
    if (!network->loadNetwork(data["clustering"]["networkPath"])) {
        return false;
    }
    else if (!network->loadCI(clusteringOptions->dataPath, clusteringOptions->startTime, clusteringOptions->endTime)) {
        return false;
    }
    */
    loaded = true;
    std::cout << "Successfully initalized gravitational clustering" << std::endl;
    return true;
}

