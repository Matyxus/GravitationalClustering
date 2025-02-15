//----------------------------------------------------------------------------------------
/**
 * \file       Renderer.hpp
 * \author     Matyáš Švadlenka
 * \contrib	   Jakub Profota
 * \date       2024/12/09
 * \brief      Class for creating window and drawing using OpenGL & SDL2
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "../utils.hpp"
#include "Shader.hpp"
#include "../network/Network.hpp"
#include "../clustering/base/Structs.hpp"
#include "../clustering/base/Options.hpp"
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <vector>


class Renderer {
public:
	/**
	  A constructor of Renderer, initializes GUI window and centers camera.

	  \param[in] limits Maximal and minimal (x, y) limits of where objects are drawn
	  \param[in] windowOptions GUI options
	*/
	Renderer(const float4 limits, const WindowOptions &windowOptions) : options(windowOptions) {
		// Center camera to middle
		camera_position = glm::ivec2(-(limits.x - limits.z) / 2.0, (limits.y - limits.w) / 2.0);
		// Zoom out so that the whole plot can be seen, zoom has to be at least 1
		mouse_wheel = std::max(std::max((limits.x - limits.z) / options.width, (limits.y - limits.w) / options.height), 1.0f);
		create_window();
	};
	~Renderer() { freeMemory(); };
	// ----------- Plotting ----------- 
	bool plotPlanets(State &state, const float multiplier);
	bool plotNetwork(Network& network);
	bool plotHeatMap(Network& network);
	bool plotClusters(Network& network, std::vector<std::vector<int>>& clusters);
	// ----------- Utils -----------
	/**
	  Polls current events from GUI, returning the even value.

	  \param[in] currentEvent Current event recorded on previous call.
	  \return Current event, based on previous.
	*/
	int pollEvents(const int currentEvent);
	inline bool isRunning() const { return running; };
private:
	// ----- Method ----- 
	/**
	  Creates OpenGL window with SDL2.

	  \return True on success, false otherwise.
	*/
	bool create_window();
	/**
	  Creates Shaders structure for Clustering and Network rendering.

	  \return True on success, false otherwise.
	*/
	bool create_shaders();
	/**
	  Computs color for heatmap from congestion index.

	  \param[in] congestionIndex Value of congestion index (0, 1)
	  \return Color between white and dark red
	*/
	inline glm::vec3 getHeatmapColor(const float congestionIndex);
	/**
	  Frees memory and destroyes OpenGL window.
	*/
	void freeMemory();
	// ----- Window objects -----
	const WindowOptions& options; ///< Options of GUI
	SDL_Window* window = nullptr; ///< OpenGL & SDL2 window
	SDL_GLContext context; ///< OpenGL & SDL2 context 
	glm::vec2 camera_position = glm::vec2(0.0f, 0.0f); ///< Curent camera positoon (x, y)
	glm::ivec2 mouse_position = glm::vec2(0, 0); ///< Curent mouse positoon (x, y)
	bool mouse_down = false; ///< Flag wheter mouse button is pressed and hold
	float mouse_wheel = 1.0f; ///< The current zoom based off mouse wheel scrolling
	bool running = false; ///< Flag wheter GUI is running
	// ----- Shaders ----- 
	Shader clusteringShader = Shader();
	Shader networkShader = Shader();
	// ----- Utils -----
	const glm::vec3 reds0 = glm::vec3(1.0f, 0.96f, 0.94f);
	const glm::vec3 reds1 = glm::vec3(0.4f, 0.0f, 0.05f);
};
