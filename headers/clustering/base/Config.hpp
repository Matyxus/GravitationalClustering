//----------------------------------------------------------------------------------------
/**
 * \file       GravClustering.h
 * \author     Matyáš Švadlenka
 * \date       2002/01/03
 * \brief      Class loading configuration file.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include <iostream>
#include <fstream>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/schema.h>
#include "Options.hpp"


/// Class that handles loading of JSON configuration file.
/**
  This class loads and stores configuration file (JSON) inside 
  structures definded in Options.hpp, provides utility methods.
*/
class Config {
public:
	Config(const std::string path) { loadConfig(path); };
	~Config();
	/**
	  Loads JSON configuration file from given path into Structs.

	  \param[in] path  full path to the JSON file.
	  \return True on success, False otherwise.
	*/
	bool loadConfig(const std::string path);
	// ----------------- Getters ----------------- 
	/**
	  \return True if configuration file is correctly loaded, false otherwise.
	*/
	inline bool isLoaded() const { return loaded; };
	/**
	  \return pointer to DeviceOptions, can be nullptr.
	*/
	inline DeviceOptions* getDeviceOptions(void) const { return deviceOptions; };
	/**
	  \return pointer to ClusteringOptions, can be nullptr.
	*/
	inline ClusteringOptions* getClusteringOptions(void) const{ return clusteringOptions; };
	/**
	  \return pointer to GuiOptions, can be nullptr.
	*/
	inline GuiOptions* getGuiOptions(void) const { return guiOptions; };
	/**
	  \return pointer to Dataptions, can be nullptr.
	*/
	inline DataOptions* getDataOptions(void) const { return dataOptions; };
private:
	// ------------------- Vars -------------------
	bool loaded = false;
	DeviceOptions* deviceOptions = nullptr;
	DataOptions* dataOptions = nullptr;
	ClusteringOptions* clusteringOptions = nullptr;
	GuiOptions* guiOptions = nullptr;
	// ------------------- Utils -------------------
	/**
		Loads JSON file to the passed document.

	    \param[in] path  full path to the JSON file.
		\param[in] doc   Document where we want to load JSON file.
		\return True if loading was successfull, false otherwise.
	*/
	bool loadJSON(const std::string path, rapidjson::Document *doc);
	/**
		Checks loaded JSON document against schema

		\param[in] schema_path   Path to schema to compare against.
		\param[in] doc   Document loaded from JSON file.
		\return True if comparison was successfull, false otherwise.
	*/
	bool checkJSON(const std::string schema_path, rapidjson::Document* doc);
};

