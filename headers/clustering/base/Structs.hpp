//----------------------------------------------------------------------------------------
/**
 * \file       Options.hpp
 * \author     Matyáš Švadlenka
 * \date       2002/01/03
 * \brief      Struct's for gravitational clustering, Grid and State.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "vector_types.h"
#include <iostream>
#include <limits>
#include <assert.h>


/// Struct for spastial structure (grid) for Gravitational Clustering.
typedef struct Grid {
	int rows = 0; ///< The number of rows
	int cols = 0; ///< The number of columns
	int size = 0; ///< Total grid size -> rows*cols
	int* cells = nullptr; ///< cells[pointID] = cellID
	int* sortedCells = nullptr; ///< sortedCells[gridMap[cellID]] = pointID
	int* gridMap = nullptr; ///< gridMap[cellID] = starting index of points in sortedCells which have cell==cellID
	int* binCounts = nullptr; ///< binCounts[cellID] = cells.count(cellID)
	int* pointOrder = nullptr; ///< pointOrder[pointID] -> order of precedence among points with the same cellID
	int neighbours[8] = { -1, 1, 0, 0, 0, 0, 0, 0 }; ///< Neighbours in all 8 directions -> cellID + neighbours[i]
	float4 borders = float4{0.f, 0.f, 0.f, 0.f}; ///< Grid coordinate limit's (max_x, max_y, min_x, min_y) 
	const bool host; ///< Flag to know where the data is allocated
	Grid(bool host) : host(host) {};
	~Grid();
	/**
	  Allocates or re-allocates (only lower size) grid based on given parameters.

	  \param[in] newBorders New grid borders (max_x, max_y, min_x, min_y) 
	  \param[in] nRows Number of rows
	  \param[in] nCols Number of cols
	  \param[in] numPoints Number of clusters
	*/
	void allocate(const float4 newBorders, const int nRows, const int nCols, const int numClusters);
	inline bool isNeigh(int cellID) const { return 0 <= cellID && cellID < size; };
	inline bool isInitialized() const { 
		return (
			rows > 0 && cols > 0 && size > 0 && 
			cells != nullptr && sortedCells != nullptr &&
			gridMap != nullptr && binCounts != nullptr && 
			pointOrder != nullptr
		); 
	};
} Grid;


/// Struct representing current State of Gravitational Clustering.
typedef struct State {
	uint32_t iteration = 0; ///< The current iteration
	int size = 0; ///< Total number of clusters currently
	int numAlive = 0; ///< Total number of clusters currently in computation
	float2* positions = nullptr; ///< Positions of clusters (x, y)
	float2* movements = nullptr; ///< Shift of cluster positions (x, y) after attraction is computed
	float* weigths = nullptr; ///< Weight of each cluster
	int2* clusters = nullptr; ///< Array mapping pointID to pair (clusterID, iteration when it merged (-1) for invalid), does not resize
	int* indexes = nullptr; ///< Indexes of each cluster, shifts after more clusters being merge
	int* merges = nullptr; ///< Array mapping clusters to other cluster which they merge with
	bool *alive = nullptr; ///< Array which tells us if given clusterID is still present in computation
	const bool host; ///< Flag to know where the data is allocated
	State(bool host) : host(host) {};
	~State();
	/**
	  Allocates or re-allocates (only lower size) state based on given parameters.

	  \param[in] numClusters Number of clusters
	*/
	void allocate(const int numClusters);
	inline bool isInitialized() const { 
		return (
			size > 0 && numAlive > 0 && positions != nullptr
			&& movements != nullptr && weigths != nullptr && 
			clusters != nullptr && indexes != nullptr && 
			merges != nullptr && alive != nullptr
		);
	}
} State;


/// Struct measuring individual functions performance.
typedef struct PeformanceRecord {
	float bordersElapsed = 0.f; ///< Total time spent on finding Grid borders
	float insertElapsed = 0.f; ///< Total time spent on inserting points
	float movementsElapsed = 0.f; ///< Total time spent on computing movements
	float neighboursElapsed = 0.f; ///< Total time spent on finding merging neighbours
	float mergingElapsed = 0.f; ///< Total time spent on merging neighbours
	float stateResizeElapsed = 0.f; ///< Total time spent on resizing state
	float gridResizeElapsed = 0.f; ///< Total time spent on resizing grid
} PeformanceRecord;

