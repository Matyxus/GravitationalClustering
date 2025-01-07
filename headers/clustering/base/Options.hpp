//----------------------------------------------------------------------------------------
/**
 * \file       Options.hpp
 * \author     Matyáš Švadlenka
 * \date       2002/01/03
 * \brief      Struct's defining options for gravitational clustering.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include <string>

/// Struct holding parameters for Device (CPU / GPU).
typedef struct DeviceOptions {
	const std::string mode; ///< Name of mode to be used, one of: [serial, parallel, GPU]
	const int threads; ///< Maximal amount threads to be used, or threads per block for GPU
	DeviceOptions(std::string mode, int threads) : mode(mode), threads(threads) {};
} DeviceOptions;


/// Struct holding parameters for Device (CPU / GPU).
typedef struct RandomOptions {
	const int size; ///< Number of elements
	const int seed; ///< Seed for RNG
	const float aliveFactor; // Chance for point to be alive (can be 1)
	const float max_x; // Maximal value x can reach
	const float max_y; // Maximal value x can reach
	RandomOptions(int size, int seed, float aliveFactor, float max_x, float max_y) :
		size(size), seed(seed), aliveFactor(aliveFactor), max_x(max_x), max_y(max_y)
	{};
} RandomOptions;

/// Struct holding parameters for resizing of State and Grid.
typedef struct ResizingOptions {
	const bool allowed; ///< Flag to tell us if resizing can be done
	const float percentage; ///< Percentage of total size current size must have (i.e. new size <= 0.5 total size -> resize)
	const int minSize; ///< Minimal amout of difference between total & new size
	const int limit; ///< Limit under which re-sizing is not done. 
	const int frequency; ///< Frequency (i.e. number of iterations) to check for Grid/State resizing (i.e. finding borders / count alive (GPU only))
	ResizingOptions(bool allowed, float percentage, int minSize, int limit, int frequency) :
		allowed(allowed), percentage(percentage), minSize(minSize), limit(limit), frequency(frequency)
	{};
} ResizingOptions;


/// Struct for plotting parameters in GravClustering class.
typedef struct PlotOptions {
	const uint16_t frequency; ///< Plotting frequency (0 for disabled)
	const uint16_t clusterSize; ///< Minimal cluster size to show
	const bool heatmap; ///< Flag to show heatmap plot
	const bool planets; ///< Flag to show planets plot
	PlotOptions(uint16_t frequency, uint16_t clusterSize, bool heatmap, bool planets) :
		frequency(frequency), clusterSize(clusterSize),
		heatmap(heatmap), planets(planets)
	{};
} PlotOptions;


/// Struct for clustering parameters in GravClustering class.
typedef struct ClusteringOptions { 
	const DeviceOptions device; ///< Options for device to be used (CPU / GPU)
	const ResizingOptions stateResize; ///< Resizing options for State
	const ResizingOptions gridResize; ///< Resizing options for Grid
	const std::string networkPath; ///< Path to SUMO's road network XML file
	const std::string dataPath; ///< Path to Congestion Index XML file
	const uint16_t iterations; ///< Number of iterations
	const float multiplier; ///< Weight multiplier
	const float startTime; ///< Congestion Index interval starting time
	const float endTime; ///< Congestion Index interval ending time
	const float radius; ///< Minimal cluster merging radius
	const float radius2; ///< Squared radius (for avoiding use of square root in distance computation)
	const float offset; ///< Offset added to all CongestionIndexes (in case some are 0)
	ClusteringOptions(
		std::string networkPath, std::string dataPath, uint16_t iterations,
		float multiplier, float startTime, float endTime, float radius, float offset, 
		const ResizingOptions stateResize, const ResizingOptions gridResize, const DeviceOptions device
	) : networkPath(networkPath), dataPath(dataPath), iterations(iterations),
		multiplier(multiplier), startTime(startTime), endTime(endTime),
		radius(radius), radius2(radius* radius), offset(offset),
		stateResize(stateResize), gridResize(gridResize), device(device)
	{};
} ClusteringOptions;

