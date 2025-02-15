//----------------------------------------------------------------------------------------
/**
 * \file       Shader.hpp
 * \author     Matyáš Švadlenka
 * \contrib	   Jakub Profota
 * \date       2024/12/09
 * \brief      Class for holding buffers of OpenGL objects.
*/
//----------------------------------------------------------------------------------------
#pragma once
#include <glm.hpp>
#include <GL/glew.h>
#include <SDL.h>
#include <iostream>

/// Struct for rendering buffers of OpenGL
typedef struct Shader {
	GLuint program = NULL; ///< The program of the shader
	GLuint vao = NULL; ///< Vertex
	GLuint vbo = NULL; ///< Buffer with attributes
	GLuint ebo = NULL; ///< Index buffer (only for network - lines)
	~Shader() {
		std::cout << "Freeing shader" << std::endl;
		if (program != NULL) {
			glDeleteProgram(program);
			program = NULL;
		}
		if (vao != NULL) {
			glDeleteVertexArrays(1, &vao);
			vao = NULL;
		}
		if (vbo != NULL) {
			glDeleteBuffers(1, &vbo);
			vbo = NULL;
		}
		if (ebo != NULL) {
			glDeleteBuffers(1, &ebo);
			ebo = NULL;
		}
	}
} Shader;

/**
  Generates OpenGL program for rendering clustering algorithm. 

  \param[in] clustering Shader for clustering
  \return True on success, false otherwise.
*/
bool generateClusteringProgram(Shader& clustering);
/**
  Generates OpenGL program for rendering road network.

  \param[in] network Shader for road network
  \return True on success, false otherwise.
*/
bool generateNetworkProgram(Shader& network);

