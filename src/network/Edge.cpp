#include "../../headers/network/Edge.hpp"


std::pair<float, float> Edge::getCentroid(void) {
    float x = 0; 
    float y = 0;
    size_t coordinates = 0;
    for (const std::vector<std::pair<float, float>>& shape : laneShapes) {
        for (const std::pair<float, float>& coord: shape) {
            x += coord.first;
            y += coord.second;
            
        }
        coordinates += shape.size();
    }
    return std::make_pair<float, float>(x / coordinates, y / coordinates);
}



std::ostream& operator<<(std::ostream& os, const Edge& edge) {
    os << "Edge: " << edge.id << "(" << edge.identifier << "), has " << edge.laneShapes.size() << " lanes";
    os << " From: " << edge.from << " to: " << edge.to;
    return os;
}


