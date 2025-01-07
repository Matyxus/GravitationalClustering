//----------------------------------------------------------------------------------------
/**
 * \file       Junction.hpp
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Class representing single Junction of road network in SUMO.
 *
 *  A more elaborated file description.
 *
*/
//----------------------------------------------------------------------------------------
#pragma once
#include <string>
#include <vector>


/// Class that described SUMO's Junction of road network.
/**
  This class represents Junction of road network defined by SUMO (https://eclipse.dev/sumo/),
  each Junction has pair of coordinates and identifiers (original and internal).
*/
class Junction {
public:
	Junction(std::string id, uint16_t identifier, float x, float y) : id(id), identifier(identifier), x(x), y(y) {};
	~Junction() {};
	const std::string id;
	const uint16_t identifier;
	const float x;
	const float y;
private:


};



