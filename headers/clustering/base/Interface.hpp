//----------------------------------------------------------------------------------------
/**
 * \file       Interface.hpp
 * \author     Matyáš Švadlenka
 * \date       2024/12/09
 * \brief      Class representing interface (base) class for Gravitational Clustering.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include <stdexcept>
#include <random>
#include "Config.hpp"
#include "Structs.hpp"
#include "../../network/Network.hpp"

/// Class that templates methods for Gravitational Clustering implementations.
/**
  This class provides template methods for subclass which implement the clustering itself.
  Also implements basic access functions for some structure and utility methods.
*/
class Interface {
public:
	virtual ~Interface() {};
	/**
	  Peforms one entire step of GravitationalClustering.

	  \return True on success, false otherwise.
	*/
	virtual bool step() = 0;
	/**
	  \return True wheter Clustering is successfully initialized, false otherwise.
	*/
	inline bool isInitialized() const { return clusteringOptions != nullptr && state.isInitialized(); };
	inline Grid &getGrid() { return grid; };
	inline State &getState() { return state; };
	inline std::vector<PeformanceRecord> &getRecords() { return records; };
protected:
	/// Constructor.
	/**
	Allocates memory and initializes variables from Network.

	\param[in] config  Configuration variables loaded from file.
	\param[in] network Road network.
	*/
	Interface(Config *config, Network *network);
	/// Constructor.
	/**
	Allocates memory and initializes variables with random numbers.

	\param[in] config  Configuration variables loaded from file.
	*/
	Interface(Config* config);
	// ------------------------ Methods ------------------------ 
	/**
	  Initializes or re-sizes Grid spatial structure.

	  \return True on success, False otherwise.

	  ERRORS:
	   - bad_alloc ... error during memory allocatation
	*/
	virtual bool generateGrid() = 0;
	/**
	  Inserts points into Grid.

	  \return True on success, False otherwise.
	*/
	virtual bool insertPoints() = 0;
	/**
	  Calculates the movements of clusters (edges) based on
	  gravitational attraction between them (all to all).

	  \return True on success, False otherwise.
	*/
	virtual bool computeMovements() = 0;
	/**
	  Find's nearest neighbour of each particle by using Grid.

	  \return True on success, false otherwise.
	*/
	virtual bool findClusters() = 0;
	/**
	  Clusterizes pairs of particles which are nearest to each other (symmetrically).

	  \return True on success, false otherwise.
	*/
	virtual bool mergeClusters() = 0;
	/**
	  Re-computed the number of alive particales, and re-sized State in case
	  the current value gets low enough (configured from JSON file).

	  \return True if re-sizing happend, false otherwise.
	*/
	virtual bool countAlive() = 0;
	/**
	  Find the current minimal and maximal (x, y) coordinates of
	  clusters, which are used as the grid limits.

	  \return Float4 containing coordinate limits (max_x, max_y, min_x, min_y)
	*/
	virtual float4 findBorders() = 0;
	// ------------------------ Utility methods ------------------------ 
	/**
	  Computes the cell in which the point is, based on its position.

	  \param[in] clusterID id of the cluster
	  \return Id of cell cluster is in.
	*/
	inline int getCell(const int clusterID) const {
		assert(clusterID < state.size);
		const int x = static_cast<int>((state.positions[clusterID].x - grid.borders.z) * clusteringOptions->params.radiusFrac);
		const int y = static_cast<int>((state.positions[clusterID].y - grid.borders.w) * clusteringOptions->params.radiusFrac);
		assert(x >= 0 && y >= 0);
		assert(0 <= x + y * grid.rows && x + y * grid.rows < grid.size);
		return x + y * grid.rows;
	}
	/**
	  Computes the Euclidean distance between two edges.

	  \param[in] edgeA id of first edge
	  \param[in] edgeB id of second edge
	  \return distance as float.
	*/
	inline float getDistance(const int edgeA, const int edgeB) const {
		const float diffX = state.positions[edgeA].x - state.positions[edgeB].x;
		const float diffY = state.positions[edgeA].y - state.positions[edgeB].y;
		return std::sqrtf(diffX * diffX + diffY * diffY);
	}
	/**
	  Computes the Squared euclidean distance between two edges.

	  \param[in] posA position of first edge
	  \param[in] posB position of second edge
	  \return distance as float.
	*/
	inline float getDistance2(const float2 &posA, const float2 &posB) const {
		const float diffX = posA.x - posB.x;
		const float diffY = posA.y - posB.y;
		return diffX * diffX + diffY * diffY;
	}
	/**
	  Checks wheter grid can be resized.

	  \param[in] newSize New size of the grid
	  \return True if grid can be resized, false otherwise.
	*/
	inline bool isGridResizable(int newSize) const {
		return (
			grid.isInitialized() && clusteringOptions->gridResize.allowed &&
			(newSize >= clusteringOptions->gridResize.limit) &&
			(newSize <= (grid.size * clusteringOptions->gridResize.percentage)) &&
			((grid.size - newSize) > clusteringOptions->gridResize.minSize)
		);
	}
	/**
	  Checks wheter State can be resized.

	  \return True if State can be resized, false otherwise.
	*/
	inline bool isStateResizable() const {
		return (
			state.isInitialized() && clusteringOptions->stateResize.allowed && 
			((state.iteration % clusteringOptions->stateResize.frequency) != 0) &&
			(state.size >= clusteringOptions->stateResize.limit) &&
			(state.numAlive <= (state.size * clusteringOptions->stateResize.percentage)) &&
			((state.size - state.numAlive) > clusteringOptions->stateResize.minSize)
		);
	}
	// ------------------------ Struct's ------------------------ 
	ClusteringOptions* clusteringOptions = nullptr; ///< pointer to ClusteringOptions (not allocated here)
	Grid grid = Grid(true); ///< Grid spatial structure, allocted by child classes
	State state = State(true); ///< State of current gravitational clustering
	PeformanceRecord record = PeformanceRecord(); ///< Struct measuring function performance
	std::vector<PeformanceRecord> records; ///< Vector holding peformance throughout iterations
};
