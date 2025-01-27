#include "../../../headers/clustering/base/Config.hpp"

Config::~Config() {
    std::cout << "Freeing Config ... " << std::endl;
    if (deviceOptions != nullptr) {
        delete deviceOptions;
        deviceOptions = nullptr;
    }
    if (dataOptions != nullptr) {
        delete dataOptions;
        dataOptions = nullptr;
    }
    if (clusteringOptions != nullptr) {
        delete clusteringOptions;
        clusteringOptions = nullptr;
    }
    if (guiOptions != nullptr) {
        delete guiOptions;
        guiOptions = nullptr;
    }
};

bool Config::loadConfig(const std::string path) {
    std::cout << "Loading json config file from: " << path << std::endl;
    // ------------------ File ------------------ 
    rapidjson::Document doc;
    if (!loadJSON(path, &doc) || ! checkJSON("./Data/config/clustering_schema.json", &doc)) {
        loaded = false;
        return false;
    }
    // ---------------- Device ---------------- 
    const rapidjson::Value& device = doc["device"];
    deviceOptions = new DeviceOptions{
        device["use"].GetString(),
        device["debug"].GetBool(),
        device["gpu_threads"].GetInt(),
        device["cpu_threads"].GetInt(),
        static_cast<uint16_t>(device["iterations"].GetUint64())
    };
    // ---------------- Data ---------------- 
    const rapidjson::Value& data = doc["data"];
    dataOptions = new DataOptions{
        data["use_rng"].GetBool(),
        NetworkOptions(
            data["network"]["path"].GetString(),
            data["network"]["edgeData"].GetString(),
            data["network"]["offset"].GetFloat(),
            data["network"]["startTime"].GetFloat(),
            data["network"]["endTime"].GetFloat()
        ),
        RandomOptions(
            data["rng"]["size"].GetInt(),
            data["rng"]["seed"].GetInt(),
            data["rng"]["aliveFactor"].GetFloat(),
            data["rng"]["max_x"].GetFloat(),
            data["rng"]["max_y"].GetFloat()
        )
    };
    // ---------------- Clustering ---------------- 
    const rapidjson::Value& clustering = doc["clustering"];
    clusteringOptions = new ClusteringOptions{
        ClusteringParams(
            clustering["params"]["multiplier"].GetFloat(),
            clustering["params"]["radius"].GetFloat()
        ),
        ResizingOptions(
            clustering["resize"]["state"]["allowed"].GetBool(),
            clustering["resize"]["state"]["percentage"].GetFloat(),
            clustering["resize"]["state"]["minSize"].GetInt(),
            clustering["resize"]["state"]["limit"].GetInt(),
            clustering["resize"]["state"]["frequency"].GetInt()
        ),
        ResizingOptions(
            clustering["resize"]["grid"]["allowed"].GetBool(),
            clustering["resize"]["grid"]["percentage"].GetFloat(),
            clustering["resize"]["grid"]["minSize"].GetInt(),
            clustering["resize"]["grid"]["limit"].GetInt(),
            clustering["resize"]["grid"]["frequency"].GetInt()
        )
    };
    // ---------------- GUI ---------------- 
    const rapidjson::Value& gui = doc["gui"];
    guiOptions = new GuiOptions{
        gui["display"].GetBool(),
        WindowOptions(
            gui["window"]["title"].GetString(),
            gui["window"]["width"].GetInt(),
            gui["window"]["height"].GetInt()
        ),
        PlotOptions(
            static_cast<uint16_t>(gui["plotting"]["frequency"].GetUint64()),
            static_cast<uint32_t>(gui["plotting"]["clusterSize"].GetUint64()),
            gui["plotting"]["heatmap"].GetBool(),
            gui["plotting"]["planets"].GetBool()
        )
    };
    // ---------------- Finished ----------------
    loaded = true;
    std::cout << "Successfully initalized gravitational clustering" << std::endl;
    return true;
}

// ------------------------------------------------------- Utils -------------------------------------------------------

bool Config::loadJSON(const std::string path, rapidjson::Document *doc) {
    assert(doc != nullptr);
    std::ifstream file(path);
    if (!file.good()) {
        std::cout << "Error: File does not exist!" << std::endl;
        loaded = false;
        return false;
    }
    rapidjson::IStreamWrapper isw{ file };
    doc->ParseStream(isw);
    if (doc->HasParseError()) {
        std::cout << "Error  : " << doc->GetParseError() << std::endl;
        std::cout << "Offset : " << doc->GetErrorOffset() << std::endl;
        return false;
    }
    return true;
}

bool Config::checkJSON(const std::string schema_path, rapidjson::Document* doc) {
    assert(doc != nullptr);
    // Load file
    std::ifstream schema_file(schema_path);
    if (!schema_file.good()) {
        std::cout << "Error: unable to load json schema: '" << schema_path << "'!" << std::endl;
        return false;
    }
    // Load Document from file
    rapidjson::Document sd;
    rapidjson::IStreamWrapper isw{ schema_file };
    sd.ParseStream(isw);
    if (sd.HasParseError()) {
        std::cout << "Error  : " << sd.GetParseError() << std::endl;
        std::cout << "Offset : " << sd.GetErrorOffset() << std::endl;
        return false;
    }
    // Compare
    rapidjson::SchemaDocument schema(sd); // Compile a Document to SchemaDocument
    rapidjson::SchemaValidator validator(schema);
    if (!doc->Accept(validator)) {
        // Input JSON is invalid according to the schema
        // Output diagnostic information
        rapidjson::StringBuffer sb;
        validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
        std::cout << "Invalid schema: " << sb.GetString() << std::endl;
        std::cout << "Invalid keyword: " << validator.GetInvalidSchemaKeyword() << std::endl;
        sb.Clear();
        validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
        std::cout << "Invalid document: " << sb.GetString() << std::endl;
        return false;
    }
    std::cout << "JSON successfully compared on schema!" << std::endl;
    return true;
}

