#include "../../headers/gui/Renderer.hpp"


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
    return create_shaders();
}

bool Renderer::create_shaders() {
    std::cout << "Initializing GUI shaders" << std::endl;
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
    if (!generateClusteringProgram(clusteringShader)) {
        return false;
    } else if (!generateNetworkProgram(networkShader)) {
        return false;
    }
    running = true;
    return true;
}


// ------------------------------- Plotting ------------------------------- 

bool Renderer::plotPlanets(State &state, const float multiplier) {
    if (!isRunning()) {
        std::cout << "Error, unable to plot planets, GUI is not running!" << std::endl;
        return false;
    }
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(clusteringShader.program);
    glBindVertexArray(clusteringShader.vao);
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    glUniform2f(glGetUniformLocation(clusteringShader.program, "resolution"), static_cast<float>(w), static_cast<float>(h));
    glUniform2f(glGetUniformLocation(clusteringShader.program, "camera_position"), -camera_position.x, camera_position.y);
    glUniform1f(glGetUniformLocation(clusteringShader.program, "camera_zoom"), mouse_wheel);
    std::vector<float> attributes; /// (x, y, size)
    attributes.reserve(state.numAlive);
    const float frac = 1.f / multiplier;
    for (int i = 0; i < state.size; i++) {
        if (state.alive[i]) {
            attributes.push_back(state.positions[i].x);
            attributes.push_back(state.positions[i].y);
            attributes.push_back(state.weigths[i] * frac);
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, clusteringShader.vbo);
    glBufferData(GL_ARRAY_BUFFER, attributes.size() * sizeof(float), attributes.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_POINTS, 0, state.numAlive);
    SDL_GL_SwapWindow(window);
    return true;
}

bool Renderer::plotNetwork(Network& network) {
    if (!isRunning()) {
        std::cout << "Error, unable to plot network, GUI is not running!" << std::endl;
        return false;
    }
    // std::cout << "Plotting heatmap!" << std::endl;
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(networkShader.program);
    glBindVertexArray(networkShader.vao);
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    glUniform2f(glGetUniformLocation(networkShader.program, "resolution"), static_cast<float>(w), static_cast<float>(h));
    glUniform2f(glGetUniformLocation(networkShader.program, "camera_position"), -camera_position.x, camera_position.y);
    glUniform1f(glGetUniformLocation(networkShader.program, "camera_zoom"), mouse_wheel);
    std::vector<float> vertices; // (x, y, R, G, B)
    std::vector<GLuint> indices; // (A -> B)
    GLuint indexOffset = 0;
    for (const auto& edge : network.getEdges()) {
        for (const auto& lane : edge->laneShapes) {
            // Position of lane points
            for (const auto& point : lane) {
                vertices.push_back(point.first);
                vertices.push_back(point.second);
                // Color per point (default is white)
                vertices.push_back(1.0f);
                vertices.push_back(1.0f);
                vertices.push_back(1.0f);
            }
            // Indexes of lane points
            for (GLuint i = 0; i < static_cast<GLuint>(lane.size() - 1); i++) {
                indices.push_back(indexOffset + i);
                indices.push_back(indexOffset + i + 1);
            }
            indexOffset += static_cast<GLuint>(lane.size());
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, networkShader.vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, networkShader.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_DYNAMIC_DRAW);
    glDrawElements(GL_LINES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
    SDL_GL_SwapWindow(window);
    return true;
    
}


bool Renderer::plotHeatMap(Network& network) {
    if (!isRunning()) {
        std::cout << "Error, unable to plot heatmap, GUI is not running!" << std::endl;
        return false;
    }
    // std::cout << "Plotting network!" << std::endl;
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(networkShader.program);
    glBindVertexArray(networkShader.vao);
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    glUniform2f(glGetUniformLocation(networkShader.program, "resolution"), static_cast<float>(w), static_cast<float>(h));
    glUniform2f(glGetUniformLocation(networkShader.program, "camera_position"), -camera_position.x, camera_position.y);
    glUniform1f(glGetUniformLocation(networkShader.program, "camera_zoom"), mouse_wheel);
    std::vector<float> vertices; // (x, y, R, G, B)
    std::vector<GLuint> indices;  // (A -> B)
    GLuint indexOffset = 0;
    glm::vec3 color;
    for (const auto& edge : network.getEdges()) {
        color = getHeatmapColor(edge->getCongestionIndex()); // Get the color from heatmap
        for (const auto& lane : edge->laneShapes) {
            // Position of lane points
            for (const auto& point : lane) {
                vertices.push_back(point.first);
                vertices.push_back(point.second);
                vertices.push_back(color.x);
                vertices.push_back(color.y);
                vertices.push_back(color.z);
            }
            // Indexes of lane points
            for (GLuint i = 0; i < static_cast<GLuint>(lane.size() - 1); i++) {
                indices.push_back(indexOffset + i);
                indices.push_back(indexOffset + i + 1);
            }
            indexOffset += static_cast<GLuint>(lane.size());
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, networkShader.vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, networkShader.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_DYNAMIC_DRAW);
    glDrawElements(GL_LINES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
    SDL_GL_SwapWindow(window);
    return true;
}

bool Renderer::plotClusters(Network& network, std::vector<std::vector<int>>& clusters) {
    if (!isRunning()) {
        std::cout << "Error, unable to plot clusters, GUI is not running!" << std::endl;
        return false;
    }
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(networkShader.program);
    glBindVertexArray(networkShader.vao);
    int w, h;
    SDL_GetWindowSize(window, &w, &h);
    glUniform2f(glGetUniformLocation(networkShader.program, "resolution"), static_cast<float>(w), static_cast<float>(h));
    glUniform2f(glGetUniformLocation(networkShader.program, "camera_position"), -camera_position.x, camera_position.y);
    glUniform1f(glGetUniformLocation(networkShader.program, "camera_zoom"), mouse_wheel);
    std::vector<float> vertices; // (x, y, R, G, B)
    std::vector<GLuint> indices; // (A -> B)
    GLuint indexOffset = 0;
    glm::vec3 color;
    for (const auto& edge : network.getEdges()) {
        color = getHeatmapColor(edge->getCongestionIndex()); // Get the color from heatmap
        for (const auto& lane : edge->laneShapes) {
            // Position of lane points
            for (const auto& point : lane) {
                vertices.push_back(point.first);
                vertices.push_back(point.second);
                vertices.push_back(color.x);
                vertices.push_back(color.y);
                vertices.push_back(color.z);
            }
            // Indexes of lane points
            for (GLuint i = 0; i < static_cast<GLuint>(lane.size() - 1); i++) {
                indices.push_back(indexOffset + i);
                indices.push_back(indexOffset + i + 1);
            }
            indexOffset += static_cast<GLuint>(lane.size());
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, networkShader.vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, networkShader.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_DYNAMIC_DRAW);
    glDrawElements(GL_LINES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
    SDL_GL_SwapWindow(window);
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
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        window = nullptr;
    }
}


inline glm::vec3 Renderer::getHeatmapColor(const float ci) {
    assert(0 <= ci && ci <= 1);
    // Define key colors from Matplotlib's "Reds" colormap
    return glm::mix(reds0, reds1, ci);
}

