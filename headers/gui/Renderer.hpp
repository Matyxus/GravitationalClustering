//----------------------------------------------------------------------------------------
/**
 * \file       Renderer.hpp
 * \author     Matyáš Švadlenka & Jakub Profota
 * \date       2024/12/09
 * \brief      Class for creating window and drawing using OpenGL & SDL2
*/
//----------------------------------------------------------------------------------------
#pragma once
#include "../utils.hpp"
#include "../network/Network.hpp"
#include "../clustering/base/Structs.hpp"
#include <glm.hpp>
#include <GL/glew.h>
#include <SDL.h>
#include <vector>

class Renderer {
	public:
		Renderer(float4 limits) {
			// Center camera to middle
			camera_position = glm::ivec2(-(limits.x - limits.z) / 2.0, (limits.y-limits.w) / 2.0);
			// Zoom out so that the whole plot can be seen
			mouse_wheel = std::max((limits.x - limits.z) / WIDTH, (limits.y - limits.w) / HEIGHT);
			create_window();
		};
		~Renderer () {
			if (window != nullptr) {
				freeMemory();
			}
		};
		// Get input
		int pollEvents(int current);
		// Draw
		bool plotPlanets(State &state, const float multiplier);
		bool plotNetwork(Network* network);
		bool plotHeatMap(Network* network);
		bool plotClusters(Network* network, std::vector<std::vector<int>> clusters);
		// Utils
		inline bool isRunning() const { return running; };
	private:
		const uint16_t WIDTH = 800;
		const uint16_t HEIGHT = 600;
		bool create_window();
		bool create_shader();
		void freeMemory();
		// Window objects
		SDL_Window* window = nullptr;
		SDL_GLContext context;
		bool running = false;
		glm::vec2 camera_position;
		glm::ivec2 mouse_position;
		bool mouse_down = false;
		float mouse_wheel;
		GLuint program;
		GLuint vao;
		GLuint position_vbo;
		GLuint size_vbo;
};











