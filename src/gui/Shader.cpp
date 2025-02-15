#include "../../headers/gui/Shader.hpp"


// ------- Forwards -------

bool createProgram(GLuint& program, const GLchar* vertexSource, const GLchar* fragmentSource);
bool linkProgram(GLuint& program, GLuint vertexShader, GLuint fragmentShader);
bool compileShader(GLuint& shader, GLenum type, const GLchar* source);

// ---------------------------------------------- Programs ----------------------------------------------

bool generateClusteringProgram(Shader& shader) {
    const GLchar vertexShaderSource[] = R"(
        #version 430 core

        layout(location = 0) in vec2 position;
        layout(location = 1) in float size;

        uniform vec2 resolution;
        uniform vec2 camera_position;
        uniform float camera_zoom;

        void main() {
          vec2 normalized = ((position - camera_position) / resolution) * 2.0 - 1.0;
          gl_Position = vec4(normalized / camera_zoom, 0.0, 1.0);
          gl_PointSize = size / camera_zoom;
        }
      )";

    const GLchar fragmentShaderSource[] = R"(
        #version 430 core

        out vec4 FragColor;

        void main() {
          FragColor = vec4(1.0, 1.0, 1.0, 1.0);
        }
      )";

    if (!createProgram(shader.program, vertexShaderSource, fragmentShaderSource)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to create clustering program\n");
        return false;
    }
    // Vertex array
    glGenVertexArrays(1, &shader.vao);
    glBindVertexArray(shader.vao);
    // Point buffer 
    glGenBuffers(1, &shader.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, shader.vbo);
    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    // Position attrib (x, y)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    // Size attrib
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));
    return true;

}

bool generateNetworkProgram(Shader& shader) {
    const GLchar vertexShaderSource[] = R"(
        #version 430 core

        layout (location = 0) in vec2 position;
        layout (location = 1) in vec3 color;

        uniform vec2 resolution;
        uniform vec2 camera_position;
        uniform float camera_zoom;

        out vec3 fragColor;

        void main()
        {
            vec2 normalized = ((position - camera_position) / resolution) * 2.0 - 1.0;
            gl_Position = vec4(normalized / camera_zoom, 0.0, 1.0);
            fragColor = color; // Pass color to fragment shader        
        }
    )";

    const GLchar fragmentShaderSource[] = R"(
        #version 430 core

        in vec3 fragColor;
        out vec4 FragColor;

        void main()
        {
            FragColor = vec4(fragColor, 1.0);
        }
    )";

    if (!createProgram(shader.program, vertexShaderSource, fragmentShaderSource)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to create network program\n");
        return false;
    }
    // Vertex array
    glGenVertexArrays(1, &shader.vao);
    glBindVertexArray(shader.vao);
    // Line buffer
    glGenBuffers(1, &shader.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, shader.vbo);
    glBufferData(GL_ARRAY_BUFFER, 5 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    // Position attrib (x, y)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Color attrib (R, G, B)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // Line indexes
    glGenBuffers(1, &shader.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shader.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
    return true;
}


// ---------------------------------------------- Utils ----------------------------------------------

/**
  Creates SDL program from given vertex and fragment sources

  \param[in] program Id of program passed by reference
  \param[in] vertexSource Source code of vertex
  \param[in] fragmentSource Source code of fragment
  \return True on success, false otherwise.
*/
bool createProgram(GLuint& program, const GLchar* vertexSource, const GLchar* fragmentSource) {
    GLuint vertex_shader;
    GLuint fragment_shader;
    if (!compileShader(vertex_shader, GL_VERTEX_SHADER, vertexSource)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to create vertex shader\n");
        return false;
    } else if (!compileShader(fragment_shader, GL_FRAGMENT_SHADER, fragmentSource)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to create fragment shader\n");
        return false;
    } else if (!linkProgram(program, vertex_shader, fragment_shader)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to create shader program\n");
        return false;
    }
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    return true;
}

/**
  Allocates or re-allocates (only lower size) grid based on given parameters.

  \param[in] program Id of program passed by reference
  \param[in] vertexShader Id of vertex shader
  \param[in] fragmentShader Id of fragment shader
  \return True on success, false otherwise.
*/
bool linkProgram(GLuint& program, GLuint vertexShader, GLuint fragmentShader) {
    program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to link program: %s\n", infoLog);
        glDeleteProgram(program);
        return false;
    }
    return true;
}

/**
  Compiles shader using OpenGL

  \param[in] shader Shader Id passed by reference
  \param[in] type Shader type
  \param[in] source Source code of shader
  \return True on success, false otherwise.
*/
bool compileShader(GLuint& shader, GLenum type, const GLchar* source) {
    shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to compile shader: %s\n", infoLog);
        glDeleteShader(shader);
        return false;
    }
    return true;
}
