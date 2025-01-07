//----------------------------------------------------------------------------------------
/**
 * \file       GravClustering.h
 * \author     Matyáš Švadlenka
 * \date       2002/01/03
 * \brief      Class loading configuration file.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "../../../lib/json/json.hpp"
#include <iostream>
#include <fstream>
#include "Options.hpp"


/// Class that handles loading of configuration file.
/**
  This class loads and stores configuration file (JSON) inside 
  classes definded in Options.hpp, provides utility methods.
*/
class Config {
public:
	Config(const std::string path) { loadConfig(path); };
	~Config();

	/**
	  Loads JSON configuration file from given path.

	  \param[in] path  full path to the XML file.
	  \return True on success, False otherwise.
	*/
	bool loadConfig(const std::string path);
	// ----------------- Getters ----------------- 
	/**
	  \return True if configuration file is correctly loaded, false otherwise.
	*/
	inline bool isLoaded() const { return loaded; };
	/**
	  \return pointer to ClusteringOptions class, can be nullptr.
	*/
	inline ClusteringOptions* getClusteringOptions(void) const{ return clusteringOptions; };
	/**
	  \return pointer to PlotOptions class, can be nullptr.
	*/
	inline PlotOptions* getPlotOptions(void) const { return plotOptions; };
	/**
	  \return pointer to getRandomOptions class, can be nullptr.
	*/
	inline RandomOptions* getRandomOptions(void) const { return randomOptions; };
private:
	bool loaded = false;
	ClusteringOptions* clusteringOptions = nullptr;
	PlotOptions* plotOptions = nullptr;
	RandomOptions* randomOptions = nullptr;
};

