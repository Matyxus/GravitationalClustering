#include "../../headers/gui/Renderer.hpp"


// ------------------------------- Draw ------------------------------- 

bool Renderer::plotPlanets(State &state, const float multiplier) {
    if (!isRunning()) {
        std::cout << "Error, unable to plot planets, GUI is not running!" << std::endl;
        return false;
    }
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(program);
    glBindVertexArray(vao);
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    glUniform2f(glGetUniformLocation(program, "resolution"), w, h);
    glUniform2f(glGetUniformLocation(program, "camera_position"), -camera_position.x, camera_position.y);
    glUniform1f(glGetUniformLocation(program, "camera_zoom"), mouse_wheel);
    // Transform positions into vec2
    std::vector<glm::vec2> positionsGUI;
    std::vector<float> weigths;
    positionsGUI.reserve(state.numAlive);
    weigths.reserve(state.numAlive);
    const float frac = 1.f / multiplier;
    for (int i = 0; i < state.size; i++) {
        if (state.alive[i]) {
            positionsGUI.push_back(glm::vec2{state.positions[i].x, state.positions[i].y });
            weigths.push_back(state.weigths[i] * frac);
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo);
    glBufferData(GL_ARRAY_BUFFER, state.numAlive * sizeof(glm::vec2), positionsGUI.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, size_vbo);
    glBufferData(GL_ARRAY_BUFFER, state.numAlive * sizeof(float), weigths.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_POINTS, 0, state.numAlive);
    SDL_GL_SwapWindow(window);
    return true;
}

bool Renderer::plotNetwork(Network* network) {
    if (!isRunning()) {
        std::cout << "Error, unable to plot network, GUI is not running!" << std::endl;
        return false;
    }
    return true;
}


bool Renderer::plotHeatMap(Network* network) {
    if (!isRunning()) {
        std::cout << "Error, unable to plot heatmap, GUI is not running!" << std::endl;
        return false;
    }
    return true;
}

bool Renderer::plotClusters(Network* network, std::vector<std::vector< int>> clusters) {
    if (!isRunning()) {
        std::cout << "Error, unable to plot clusters, GUI is not running!" << std::endl;
        return false;
    }
    return true;
}

// ------------------------------- Init ------------------------------- 

bool Renderer::create_window() {
    std::cout << "Initializing GUI window" << std::endl;
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to initialize SDL: %s\n", SDL_GetError());
        SDL_Quit();
        return false;
    }

    window = SDL_CreateWindow(
        options.title.data(), SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, options.width, options.height,
        SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL
    );
    if (window == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to create window: %s\n", SDL_GetError());
        SDL_Quit();
        return false;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    context = SDL_GL_CreateContext(window);
    if (context == NULL) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to create OpenGL context: %s\n", SDL_GetError());
        SDL_Quit();
        return false;
    }
    SDL_GL_SetSwapInterval(1);
    return create_shader();
}
bool Renderer::create_shader() {
    std::cout << "Initializing GUI shader" << std::endl;
    GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to initialize GLEW: %s\n", glewGetErrorString(glew_status));
        SDL_Quit();
        return false;
    }
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glDisable(GL_CULL_FACE);
    glDisable(GL_STENCIL_TEST);

    const GLchar vertex_shader[] = R"(
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

    const GLchar fragment_shader[] = R"(
        #version 430 core

        out vec4 frag_color;

        void main() {
          float d = distance(gl_PointCoord, vec2(0.5, 0.5));
          if (d > 0.5) discard;
          else
            frag_color = vec4(1.0, 1.0, 1.0, 1.0);
        }
      )";

    GLuint vertex_shader_id = glCreateShader(GL_VERTEX_SHADER);
    const GLchar* vertex_shader_ptr = vertex_shader;
    glShaderSource(vertex_shader_id, 1, &vertex_shader_ptr, NULL);
    glCompileShader(vertex_shader_id);

    GLuint fragment_shader_id = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar* fragment_shader_ptr = fragment_shader;
    glShaderSource(fragment_shader_id, 1, &fragment_shader_ptr, NULL);
    glCompileShader(fragment_shader_id);

    program = glCreateProgram();
    glAttachShader(program, vertex_shader_id);
    glAttachShader(program, fragment_shader_id);
    glLinkProgram(program);

    glDeleteShader(vertex_shader_id);
    glDeleteShader(fragment_shader_id);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &position_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, position_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &size_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, size_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), 0);
    glEnableVertexAttribArray(1);
    running = true;
    return true;
}


// ------------------------------- Utils ------------------------------- 

int Renderer::pollEvents(const int currentEvent) {
    if (!isRunning()) {
        std::cout << "Error, unable to poll events, GUI is not running!" << std::endl;
        return QUIT;
    }
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            // ---------------- Quit ---------------- 
            case SDL_QUIT:
                std::cout << "Closing window!" << std::endl;
                freeMemory();
                return QUIT;
            // ---------------- Keyboard ---------------- 
            case SDL_KEYDOWN:
                /* Check the SDLKey values and move change the coords */
                switch (event.key.keysym.sym) {
                    case SDLK_SPACE:
                        std::cout << "Pressed: 'space bar' !" << std::endl;
                        return (currentEvent == PAUSE) ? RUN : PAUSE;
                    case SDLK_LEFT:
                        std::cout << "Pressed: '<-'!" << std::endl;
                        return PREVIOUS;
                    case SDLK_RIGHT:
                        std::cout << "Pressed: '->' !" << std::endl;
                        return NEXT;
                    default:
                        break;
                }
                break;
            // ---------------- Mouse ---------------- 
            case SDL_MOUSEBUTTONDOWN:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    mouse_down = true;
                    mouse_position = glm::ivec2(event.button.x, event.button.y);
                }
                break;
            case SDL_MOUSEBUTTONUP:
                mouse_down = !(event.button.button == SDL_BUTTON_LEFT);
                break;
            case SDL_MOUSEMOTION:
                if (mouse_down) {
                    glm::ivec2 delta(event.motion.x, event.motion.y);
                    delta -= mouse_position;
                    mouse_position = glm::ivec2(event.motion.x, event.motion.y);
                    camera_position += glm::vec2(delta);
                }
                break;
            case SDL_MOUSEWHEEL:
                if (event.wheel.y > 0) {
                    mouse_wheel /= 1.025f;
                } else if (event.wheel.y < 0) {
                    mouse_wheel *= 1.025f;
                }
                break;
            default:
                break;
        }
    }
    return currentEvent;
}

void Renderer::freeMemory() {
    running = false;
    if (window != nullptr) {
        std::cout << "Freeing GUI" << std::endl;
        glDeleteProgram(program);
        glDeleteBuffers(1, &size_vbo);
        glDeleteVertexArrays(1, &vao);
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        window = nullptr;
    }
}

