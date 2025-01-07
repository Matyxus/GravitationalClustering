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

	  \return True on success, False otherwise.
	*/
	virtual bool step() = 0;
	inline bool isInitialized() const { return plotOptions != nullptr && clusteringOptions != nullptr; };
	inline Grid &getGrid() { return grid; };
	inline State &getState() { return state; };
protected:
	/// Constructor.
	/**
	Assigns pointers and initializes variables.

	\param[in] config  Configuration variables loaded from file.
	\param[in] network Road network.
	*/
	Interface(Config *config, Network *network);
	/// Constructor.
	/**
	Assigns pointers and initializes variables with random numbers.

	\param[in] config  Configuration variables loaded from file.
	\param[in] randomOptions Options for RNG
	*/
	Interface(Config* config, RandomOptions* randomOptions);
	// ------------------------ Methods ------------------------ 
	/**
	  Initializes all variables and arrays for gravitational clustering.

	  \param[in] network Road network loaded from file.
	  \return True on success, False otherwise.
	*/
	virtual bool initialize(Network* network) = 0;
	/**
	  Initializes all variables and arrays for gravitational clustering with random numbers.

	  \param[in] randomOptions Options for RNG
	  \return True on success, False otherwise.
	*/
	virtual bool initialize(RandomOptions* randomOptions) = 0;
	/**
	  Initializes Grid spatial structure.

	  \param[in] re_size True if existing grid should be reallocated (decreased in size).
	  \return True on success, False otherwise.

	  ERRORS:
	   - bad_alloc ... error during memory allocatation
	*/
	virtual bool generateGrid(bool re_size) = 0;
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

	  \return True on success, False otherwise.
	*/
	virtual bool findClusters() = 0;
	/**
	  Clusterizes pairs of particles which are nearest to each other (symmetrically).

	  \return True on success, False otherwise.
	*/
	virtual bool mergeClusters() = 0;
	/**
	  Resizes the arrays in case number of "alive" points gets low enough.

	  Depends on configured parameters (usually half).

	  \return True if re-sizing happend, false otherwise.

	  ERRORS:
		- bad_alloc ... error during memory allocatation
	*/
	virtual bool reSize() = 0;
	/**
	  Find the current minimal and maximal (x, y) coordinates of
	  clusters (edges), which are used as the grid limits.

	  \return Float4 containing coordinate limit's (max_x, max_y, min_x, min_y)
	*/
	virtual float4 findBorders() = 0;
	// ------------------------ Utility methods ------------------------ 
	/**
	  Generates random values to State, based on given parameters.

	  \return True on success, false otherwise.
	*/
	bool generateRandomValues();
	/**
	  Checks Grid structure correctness.

	  \return True on success, false otherwise.
	*/
	bool checkGrid();
	/**
	  Computes the cell on which the point is, based on its position.

	  \param[in] edgeID id of the edge
	  \return Id of cell edge is on.
	*/
	inline int getCell(const int edgeID) const {
		assert(edgeID < state.numEdges);
		const int x = static_cast<int>((state.positions[edgeID].x - grid.borders.z) / clusteringOptions->radius);
		const int y = static_cast<int>((state.positions[edgeID].y - grid.borders.w) / clusteringOptions->radius);
		if (!(x >= 0 && y >= 0)) {
			std::cout << "EdgeID: " << edgeID << std::endl;
			std::cout << "Pos.x: " << state.positions[edgeID].x << ", Pos.y: " << state.positions[edgeID].y << std::endl;
			std::cout << "Grid.z: " << grid.borders.z << ", Grid.w: " << grid.borders.w << std::endl;
			std::cout << "x,y = " << x << " - " << y << std::endl;
		}
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
			state.isInitialized() && clusteringOptions->gridResize.allowed &&
			// ((state.iteration % clusteringOptions->stateResize.frequency) == 0) &&
			(newSize >= clusteringOptions->gridResize.limit) &&
			(newSize <= (grid.size * clusteringOptions->gridResize.percentage)) &&
			((grid.size - newSize) < clusteringOptions->gridResize.minSize)
		);
	}
	/**
	  Checks wheter State can be resized.

	  \return True if State can be resized, false otherwise.
	*/
	inline bool isStateResizable() const {
		return (
			state.isInitialized() && clusteringOptions->stateResize.allowed && 
			// ((state.iteration % clusteringOptions->stateResize.frequency) == 0) &&
			(state.numEdges >= clusteringOptions->stateResize.limit) &&
			(state.numAlive <= (state.numEdges * clusteringOptions->stateResize.percentage)) &&
			((state.numEdges - state.numAlive) > clusteringOptions->stateResize.minSize)
		);
	}

	// ------------------------ Struct's ------------------------ 
	PlotOptions* plotOptions = nullptr; ///< pointer to PlottingOptions (not allocated here)
	ClusteringOptions* clusteringOptions = nullptr; ///< pointer to ClusteringOptions (not allocated here)
	RandomOptions* randomOptions = nullptr; ///< pointer to RandomOptions (not allocated here)
	Network* network = nullptr; ///< pointer to Network (not allocated here)
	Grid grid = Grid(true); ///< Grid spatial structure, allocted by child classes
	State state = State(true); ///< State of current gravitational clustering
};




