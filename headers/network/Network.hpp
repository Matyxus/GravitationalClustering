//----------------------------------------------------------------------------------------
/**
 * \file       Network.hpp
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Class representing road network in SUMO.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "pugixml.hpp"
#include "Junction.hpp"
#include "Edge.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <limits>
#include <unordered_map>

/// Class that described SUMO's road network.
/**
  This class represents road network defined by SUMO (https://eclipse.dev/sumo/),
  contains Edges and Junctions (each represented by custom Class) with their attributes,
  along with utility methods.
*/
class Network {

public:
	/// A constructor. Initializes class variables.
	Network() {};
	Network(const std::string networkPath) { loadNetwork(networkPath); };
	Network(const std::string networkPath, const std::string dataPath) { loadNetwork(networkPath); loadCI(dataPath); };
	Network(
		const std::string networkPath, const std::string dataPath, 
		const float startTime, const float endTime, const float offset) { 
		loadNetwork(networkPath); 
		loadCI(dataPath, startTime, endTime, offset); 
	};

	/// A destructor. Free's memory of class variables.
	~Network() {
		std::cout << "Freeing Network" << std::endl;
		for (Edge *edge: edges) {
			delete edge;
		}
		edges.clear();
		for (Junction* junction : junctions) {
			delete junction;
		}
		junctions.clear();
	}
	
	// ----------------------------------------- Load -----------------------------------------

	/**
	  Loads road networks objects (edges & junctions) with their
	  attributes from XML file used by SUMO.

	  \param[in] path  full path to the road network file.
	  \return True on success, False otherwise.
	*/
	bool loadNetwork(std::string path);

	/**
	  Loads congestion indexes from XML file for current network edges.

	  \param[in] path  full path to the XML file.
	  \param[in] startTime  starting time of interval to be taken from data.
	  \param[in] endTime  ending time of interval to be taken from data.
	  \param[in] offset Value added to congestions which are 0. (default 0.01)
	  \return True on success, False otherwise.
	*/
	bool loadCI(
		const std::string path, const float startTime = 0., 
		const float endTime = std::numeric_limits<float>::infinity(), const float offset = 0.01f
	);


	// ----------------------------------------- Exists ----------------------------------------- 

	/**
	  Helper function which checks if the id of an Edge is valid.

	  \param[in] id  identifier of the Edge (string for original, uint16_t for internal).
	  \return True if Edge exists, false otherwise.
	*/
	inline bool edgeExists(std::string id) { return edgeMap.count(id) != 0; };
	inline bool edgeExists(uint16_t id) { return id < edges.size(); };

	/**
	  Helper function which checks if the id of an Junction is valid.

	  \param[in] id  identifier of the Junction (string for original, uint16_t for internal).
	  \return True if Junction exists, false otherwise.
	*/
	inline bool junctionExists(std::string id) { return junctionMap.count(id) != 0; };
	inline bool junctionExists(uint16_t id) { return id < junctions.size(); };

	// ----------------------------------------- Getters ----------------------------------------- 

	/**
	Helper function for getting Edges based on their id.

	  \param[in] id  identifier of the Edge (string for original, uint16_t for internal).
	  \return Edge if it exists based on id, nullptr otherwise.
	*/
	inline Edge* getEdge(std::string id) { return edgeExists(id) ? edges[edgeMap[id]] : nullptr; }
	inline Edge* getEdge(uint16_t id) { return edgeExists(id) ? edges[id] : nullptr; };
	inline std::vector<Edge*> &getEdges(void) { return edges; };

	/**
	Helper function for getting Junctions based on their id.

	  \param[in] id  identifier of the Junction (string for original, uint16_t for internal).
	  \return Junction if it exists based on id, nullptr otherwise.
	*/
	inline Junction* getJunction(std::string id) { return junctionExists(id) ? junctions[junctionMap[id]] : nullptr; };
	inline Junction* getJunction(uint16_t id) { return junctionExists(id) ? junctions[id] : nullptr; };
	inline std::vector<Junction*>& getJunctions(void) { return junctions; };


private:
	// ----------------------------------------- Vars ----------------------------------------- 

	std::vector<Junction*> junctions; ///< Vector of Junctions of road networks (pointers)
	std::vector<Edge*> edges; ///< Vector of Edges of road networks (pointers)
	std::unordered_map<std::string, uint16_t> edgeMap;	///< Map of edge id's to their internal id's
	std::unordered_map<std::string, uint16_t> junctionMap; ///< Map of junction id's to their internal id's

	// ----------------------------------------- Loading ----------------------------------------- 

	/**
	  Loads road network edges with their attributes from XML file used by SUMO.

	  \param[in] doc  document of the opened XML road network file.
	  \return True on success, False otherwise.
	*/
	bool loadEdges(pugi::xml_document &doc);

	/**
	  Loads road network junctions with their attributes from XML file used by SUMO.

	  \param[in] doc  document of the opened XML road network file.
	  \return True on success, False otherwise.
	*/
	bool loadJunctions(pugi::xml_document &doc);

	/**
	  Helper function which tells if the road network was loaded.

	  \return True if edges and junctions are not empty, False otherwise.
	*/
	inline bool isLoaded(void) { return !(junctions.empty() || edges.empty()); };

	// ----------------------------------------- Utils ----------------------------------------- 

	/**
	Helper function which splits string by given delimiter.

	  \return Vector of strings split by delimiter.
	*/
	std::vector<std::string> split(const std::string& s, const std::string& delimiter);

	/**
	Helper function which checks the XML document validity and name of first element.

	  \return True if everything checks out, False otherwise.
	*/
	bool checkDoc(pugi::xml_document& doc, std::string firstName);
};


