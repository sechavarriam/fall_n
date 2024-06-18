#ifndef ADJACENCYLIST_HH
#define ADJACENCYLIST_HH

#include <vector>

class AdjacencyList { // REVISAR! 
private:
    std::vector<std::vector<int>> adjList;

public:
    AdjacencyList(int numVertices) : adjList(numVertices) {}

    void addEdge(int src, int dest) {
        adjList[src].push_back(dest);
        // If the graph is undirected, uncomment the line below
        // adjList[dest].push_back(src);
    }

    const std::vector<int>& getNeighbors(int vertex) const {
        return adjList[vertex];
    }
};

#endif // ADJACENCYLIST_HH