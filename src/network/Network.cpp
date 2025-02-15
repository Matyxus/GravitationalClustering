#include "../../headers/network/Network.hpp"


// --------------------------------------- Loading Network --------------------------------------- 

bool Network::loadNetwork(std::string path) {
	// Load XML File
	std::cout << "Loading road network from: " << path << std::endl;
	pugi::xml_document doc;
	if (!doc.load_file(path.c_str())) {
		std::cout << "Error at opening file: " << path << " !" << std::endl;
		return false;
	// Read file and load network components
	} else if (!loadEdges(doc) || !loadJunctions(doc)) {
		return false;
	}
	std::cout << "Successfully loaded network." << std::endl;
	return true;
}

bool Network::loadEdges(pugi::xml_document &doc) {
	// Check XML doc
	if (!checkDoc(doc, "net")) {
		return false;
	}
	// Find non-internal edges and load their attributes
	uint16_t num_edges = 0;
	std::vector<std::vector<std::pair<float, float>>> laneShapes;
	for (pugi::xml_node xml_edge : doc.first_child().children("edge")) {
		if (xml_edge.attribute("id").as_string()[0] == ':') {
			continue;
		}
		// Load Edge Lanes
		for (pugi::xml_node xml_lane : xml_edge.children()) {
			// Parse shape into vector of pairs (x and y coordinates)
			std::vector<std::pair<float, float>> shape;
			for (const auto& coords : split(xml_lane.attribute("shape").as_string(), " ")) {
				const std::vector<std::string>& coord = split(coords, ",");
				shape.push_back({ std::stof(coord[0]), std::stof(coord[1]) });
			}
			laneShapes.push_back(shape);
		}
		Edge *edge = new Edge(
			(std::string) xml_edge.attribute("id").as_string(), num_edges, 
			(std::string) xml_edge.attribute("from").as_string(), 
			(std::string) xml_edge.attribute("to").as_string(), laneShapes
		);
		edges.push_back(edge);
		edgeMap[edge->id] = edge->identifier;
		num_edges++;
		laneShapes.clear();
 	}
	std::cout << "Successfully loaded " << num_edges <<  " edges." << std::endl;
	return (num_edges != 0);
}

bool Network::loadJunctions(pugi::xml_document &doc) {
	// Check XML doc
	if (!checkDoc(doc, "net")) {
		return false;
	}
	// Find non-internal junctions and load their attributes
	uint16_t num_junctions = 0;
	for (pugi::xml_node xml_junction : doc.first_child().children("junction")) {
		if (xml_junction.attribute("id").as_string()[0] == ':') {
			continue;
		}
		Junction* junction = new Junction(
			(std::string)xml_junction.attribute("id").as_string(), num_junctions,
			xml_junction.attribute("x").as_float(), xml_junction.attribute("y").as_float()
		);
		junctions.push_back(junction);
		junctionMap[junction->id] = junction->identifier;
		num_junctions++;
	}
	std::cout << "Successfully loaded " << num_junctions << " junctions." << std::endl;
	return (num_junctions != 0);
}

// --------------------------------------- Loading CI --------------------------------------- 

bool Network::loadCI(const std::string path, const float startTime, const float endTime, const float offset) {
	// Check if the network was loaded
	if (!isLoaded()) {
		std::cout << "Error, Network has to be loaded before CongestionIndexes!" << std::endl;
		return false;
	// Check params
	} else if (startTime < 0) {
		std::cout << "Error, parameter 'startTime' cannot be lower than 0!" << std::endl;
		return false;
	} else if (startTime >= endTime) {
		std::cout << "Error, parameter 'startTime' cannot be bigger than 'endTime'!" << std::endl;
		return false;
	}
	// Load XML File
	std::cout << "Loading CongestionIndexes from: " << path << std::endl;
	pugi::xml_document doc;
	if (!doc.load_file(path.c_str())) {
		std::cout << "Error at opening file: " << path << " !" << std::endl;
		return false;
	} else if (!checkDoc(doc, "meandata")) {
		return false;
	}
	// Read file and load CI to edges of network
	uint16_t num_intervals = 0;
	std::vector<std::pair<float, uint16_t>> congestions(edges.size(), std::make_pair(0.f, 0)); // (CI, count)
	uint16_t id = 0;
	for (pugi::xml_node xml_interval : doc.first_child().children("interval")) {
		// Load data only between given time intervals
		if (xml_interval.attribute("begin").as_float() < startTime) {
			continue;
		} else if (xml_interval.attribute("end").as_float() > endTime) {
			break;
		}
		// Load data
		for (pugi::xml_node xml_edge: xml_interval.children()) {
			// Check validity
			if (!xml_edge.attribute("congestionIndex")) {
				std::cout << "Error, edge: '" << xml_edge.attribute("id").as_string()  << "' is mising attribute 'congestionIndex' !" << std::endl;
				return false;
			} else if (!edgeExists(xml_edge.attribute("id").as_string())) {
				// std::cout << "Warning, edge: '" << xml_edge.attribute("id").as_string() << "' does not exist!" << std::endl;
				continue;
			} else { // Load value
				id = edgeMap[xml_edge.attribute("id").as_string()];
				congestions[id].first += xml_edge.attribute("congestionIndex").as_float();
				congestions[id].second++;
			}
		}
		num_intervals++;
	}
	// Assign averaged values to edges
	uint32_t setCongestions = 0;
	for (size_t i = 0; i < edges.size(); i++) {
		if (congestions[i].second == 0) {
			edges[i]->setCongestionIndex(offset);
			setCongestions++;
		} else {
			edges[i]->setCongestionIndex(congestions[i].first / congestions[i].second);
		}
	}
	if (setCongestions != 0) {
		std::cout << "Filled: " << setCongestions << "/" << edges.size() << " edges with default offset: " << offset << std::endl;
	}
	std::cout << "Successfully loaded " << num_intervals << " CI intervals." << std::endl;
	return true;
}


// --------------------------------------- Utils --------------------------------------- 

bool Network::checkDoc(pugi::xml_document& doc, std::string firstName) {
	if (doc == nullptr || doc.empty() || doc.first_child() == nullptr || doc.first_child().empty()) {
		std::cout << "Error at reading xml file, document is empty!" << std::endl;
		return false;
	} else if (strcmp(doc.first_child().name(), firstName.data()) != 0) {
		std::cout << "Error at reading xml doc, expected different format!" << std::endl;
		return false;
	}
	return true;
}

std::vector<std::string> Network::split(const std::string& s, const std::string& delimiter) {
	size_t pos_start = 0, pos_end, delim_len = delimiter.length();
	std::string token;
	std::vector<std::string> res;
	while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
		token = s.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		res.push_back(token);
	}
	res.push_back(s.substr(pos_start));
	return res;
}
