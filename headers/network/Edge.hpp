//----------------------------------------------------------------------------------------
/**
 * \file       Edge.hpp
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Class representing single Edge (road segment) of road network in SUMO.
 *
 *  A more elaborated file description.
 *
*/
//----------------------------------------------------------------------------------------
#pragma once
#include <string>
#include <vector>
#include <iostream>

/// Class that described SUMO's Edge of road network.
/**
  This class represents Edge of road network defined by SUMO (https://eclipse.dev/sumo/),
  each Edge has lines (or single), which are for simplicity only represented by their shape as 
  vecotr of coordinate pairs.
*/
class Edge {
public:
	Edge(
		std::string id, uint16_t identifier, 
		std::string from, std::string to,
		std::vector<std::vector<std::pair<float, float>>>& laneShapes
	) : id(id), identifier(identifier), from(from), to(to), laneShapes(laneShapes)
	{};
	~Edge() {};

	const std::string id; ///< Original id (from file)
	const uint16_t identifier; ///< Internal id
	const std::string from; ///< Junction id (original)
	const std::string to; ///< Junction id (original)
	const std::vector<std::vector<std::pair<float, float>>> laneShapes; ///< Shapes of all lanes, saved as (x, y) coordinates in vectors

	inline void setCongestionIndex(float value) {
		if (value < 0 || value > 1) {
			std::cout << "Error, cannot set congestion index as: " << value << " it has to be in interval <0, 1> !" << std::endl;
			return;
		}
		congestionIndex = value;
	}
	inline float getCongestionIndex(void) { return congestionIndex; };

	std::pair<float, float> getCentroid(void);

	// Declare the friend function for operator<<
	friend std::ostream& operator<<(std::ostream& os, const Edge& edge);
	
private:
	float congestionIndex = 0; // congestion index of edge
	

};









