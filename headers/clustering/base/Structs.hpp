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
	// Utils
	const bool host; ///< Flag to know where the data is allocated
	int padding = 0; ///< Padding added to arrays (GPU only to to make sizes 2^x)
	Grid(bool host) : host(host) {};
	~Grid() {
		if (!host) {
			return;
		}
		std::cout << "Freeing Grid on CPU" << std::endl;
		if (cells != nullptr) {
			free(cells);
			cells = nullptr;
		}
		if (sortedCells != nullptr) {
			free(sortedCells);
			sortedCells = nullptr;
		}
		if (gridMap != nullptr) {
			free(gridMap);
			gridMap = nullptr;
		}
		if (binCounts != nullptr) {
			free(binCounts);
			binCounts = nullptr;
		}
		if (pointOrder != nullptr) {
			free(pointOrder);
			pointOrder = nullptr;
		}
	}
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
	uint16_t iteration = 0; ///< The current iteration
	int numEdges = 0; ///< Total number of edges/clusters currently
	int numAlive = 0; ///< Total number of edges/clusters currently in computation
	float2* positions = nullptr; ///< Positions of clusters (x, y)
	float2* movements = nullptr; ///< Shift of cluster positions (x, y) after attraction is computed
	float* weigths = nullptr; ///< Weight of each cluster
	int2* clusters = nullptr; ///< Array mapping pointID to (clusterID, iteration when it merged (-1) for invalid), never resizes!
	int* indexes = nullptr; ///< Indexes of each cluster, shifts after more clusters being merge
	int* merges = nullptr; ///< Array mapping clusters to other cluster which they merge with
	bool *alive = nullptr; ///< Array which tells us if given clusterID is still present in computation
	// Utils
	const bool host; ///< Flag to know where the data is allocated
	State(bool host) : host(host) {};
	~State() {
		if (!host) {
			return;
		}
		std::cout << "Freeing State on CPU" << std::endl;
		if (positions != nullptr) {
			free(positions);
			positions = nullptr;
		}
		if (movements != nullptr) {
			free(movements);
			movements = nullptr;
		}
		if (weigths != nullptr) {
			free(weigths);
			weigths = nullptr;
		}
		if (indexes != nullptr) {
			free(indexes);
			indexes = nullptr;
		}
		if (merges != nullptr) {
			free(merges);
			merges = nullptr;
		}
		if (alive != nullptr) {
			free(alive);
			alive = nullptr;
		}
		if (clusters != nullptr) {
			free(clusters);
			clusters = nullptr;
		}
	}
	inline bool isInitialized() const { 
		return (
			numEdges > 0 && numAlive > 0 && positions != nullptr 
			&& movements != nullptr && weigths != nullptr && 
			clusters != nullptr && indexes != nullptr && 
			merges != nullptr && alive != nullptr
		);
	}
} State;
