//----------------------------------------------------------------------------------------
/**
 * \file       Options.hpp
 * \author     Matyáš Švadlenka
 * \date       2002/01/03
 * \brief      Struct's defining options of program (Device, Data, Clustering, GUI).
*/
//----------------------------------------------------------------------------------------
#pragma once
#include <string>


// ----------------------------- Device -----------------------------

/// Struct holding parameters for Device (CPU / GPU) and Logging.
typedef struct DeviceOptions {
	const std::string use; ///< Name of device to be used, one of: [cpu, gpu]
	const bool debug; ///< Flag to determine wheter "debug" mode is used (Logging).
	const int gpu_threads; ///< Maximal amount threads threads per block for GPU
	const int cpu_threads; ///< Maximal amount of threads to be used for parallel CPU (serial mode if only 1)
	const uint16_t iterations; ///< Number of iterations
	DeviceOptions(std::string use, bool debug, int gpu_threads, int cpu_threads, uint16_t iterations) :
		use(use), debug(debug), gpu_threads(gpu_threads), cpu_threads(cpu_threads), iterations(iterations)
	{};
} DeviceOptions;


// ----------------------------- Data -----------------------------

/// Struct holding parameters for randomly generated data.
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

/// Struct holding parameters for SUMO's network data.
typedef struct NetworkOptions {
	const std::string path; ///< File path to SUMO's network file (net.xml)
	const std::string edgeData; ///< File path to SUMO's edge dump file (out.xml)
	const float offset; ///< Offset added to all CongestionIndexes (in case some are 0)
	const float startTime; ///< Congestion Index interval starting time
	const float endTime; ///< Congestion Index interval ending time
	NetworkOptions(std::string path, std::string edgeData, float offset, float startTime, float endTime) :
		path(path), edgeData(edgeData), offset(offset), startTime(startTime), endTime(endTime)
	{};
} NetworkOptions;


/// Struct holding data used by the algorithm.
typedef struct DataOptions {
	const bool use_rng; ///< True if Random data should be use, otherwise Network will be used.
	const NetworkOptions networkOptions; ///< Options for network data
	const RandomOptions randomOptions; ///< Options for random data
	DataOptions(bool use_rng, NetworkOptions networkOptions, RandomOptions randomOptions) :
		use_rng(use_rng), networkOptions(networkOptions), randomOptions(randomOptions)
	{};
} DataOptions;


// ----------------------------- Clustering -----------------------------

/// Struct holding parameters for resizing of State and Grid.
typedef struct ResizingOptions {
	const bool allowed; ///< Flag to tell us if resizing can be done
	const float percentage; ///< Percentage of total size current size must have (i.e. (new size <= 0.5 total size) -> resize)
	const int minSize; ///< Minimal amout of difference between total & new size
	const int limit; ///< Limit of size under which re-sizing is not done. 
	const int frequency; ///< How often should Grid/State resizing be checked for (i.e. finding borders / count alive (GPU only))
	ResizingOptions(bool allowed, float percentage, int minSize, int limit, int frequency) :
		allowed(allowed), percentage(percentage), minSize(minSize), limit(limit), frequency(frequency)
	{};
} ResizingOptions;


/// Struct holding parameters of clustering.
typedef struct ClusteringParams{
	const float multiplier; ///< Weight multiplier
	const float radius; ///< Minimal cluster merging radius
	const float radius2; ///< Squared radius (for avoiding use of square root in distance computation)
	const float radiusFrac; /// Fraction of radius -> 1/radius (to replace division by multiplication)
	ClusteringParams(float multiplier, float radius) :
		multiplier(multiplier), radius(radius), radius2(radius * radius), radiusFrac((float)(1.f / radius))
	{};
} ClusteringParams;


/// Struct for clustering options.
typedef struct ClusteringOptions {
	const ClusteringParams params; ///< Parameters of clustering
	const ResizingOptions stateResize; ///< Resizing options for State
	const ResizingOptions gridResize; ///< Resizing options for Grid
	ClusteringOptions(ClusteringParams params, ResizingOptions stateResize, ResizingOptions gridResize) :
		params(params), stateResize(stateResize), gridResize(gridResize)
	{};
} ClusteringOptions;

// ----------------------------- GUI -----------------------------

/// Struct holding parameters for GUI window.
typedef struct WindowOptions {
	const std::string title; ///< Title (name) of the window
	const uint16_t width; ///< Width of window (in pixels)
	const uint16_t height; ///< Height of window (in pixels)
	WindowOptions(std::string title, uint16_t width, uint16_t height) : title(title), width(width), height(height) {};
} WindowOptions;


/// Struct holding parameters of plotting
typedef struct PlotOptions {
	const uint16_t frequency; ///< Plotting frequency (0 for disabled)
	const uint32_t clusterSize; ///< Minimal cluster size to show
	const bool heatmap; ///< Flag to show heatmap plot
	const bool planets; ///< Flag to show planets plot
	PlotOptions(uint16_t frequency, uint32_t clusterSize, bool heatmap, bool planets) :
		frequency(frequency), clusterSize(clusterSize),
		heatmap(heatmap), planets(planets)
	{};
} PlotOptions;


/// Struct holding parameters of GUI (rendering & window)
typedef struct GuiOptions {
	const bool display; ///< Flag denoting wheter visual output is done
	const WindowOptions windowOptions; ///< GUI window options
	const PlotOptions plotOptions; ///< Plotting options of GUI
	GuiOptions(bool display, WindowOptions windowOptions, PlotOptions plotOptions) :
		display(display), windowOptions(windowOptions), plotOptions(plotOptions)
	{};
} GuiOptions;

