#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <stack>

template <typename Type>
using V1 = std::vector<Type>;
template <typename Type>
using V2 = std::vector<V1<Type>>;

namespace Utility
{
    V2<int> adjListToMatrix(const V2<int> &adjList)
    {
        const int len = adjList.size();
        V2<int> matrix(len, V1<int>(len, 0));

        for (int i = 1; i < len; ++i)
            for (int j = 1; j < len; ++j)
                matrix[i][j] = 1;
        return matrix;
    }
    V2<int> adjMatrixToList(const V2<int> &adjMatrix)
    {
        const int len = adjMatrix.size();
        V2<int> adjList(len);

        for (int i = 1; i < len; ++i)
            for (int j = 1; j < len; ++j)
                adjList[i].push_back(j);
        return adjList;
    }
} // utility
namespace IntroToGraph
{
    // adjacency list: a way to store graph edjes and nodes
    // adjacency matrix: store the graph in a 2d array
    void BFSdemo(const V2<int> &adjList)
    {
        V1<bool> vis(adjList.size(), 0);
        if (adjList.empty())
            return;
        std::queue<int> q;
        q.push(1);
        vis[1] = true;
        std::cout << 1;

        while (!q.empty())
        {
            int nodeIdx = q.front();
            q.pop();

            for (int node : adjList[nodeIdx])
            {
                if (vis[node])
                    continue;
                vis[node] = true;
                std::cout << node;
                q.push(node);
            }
        }
    }
    void DFSdemo(const V2<int> &adjMatrix, int node, V1<bool> &vis)
    {
        if (vis[node])
            return;
        std::cout << node << " ";
        vis[node] = true;

        for (int j = 1; j < adjMatrix.size(); ++j)
        {
            if (adjMatrix[node][j])
                DFSdemo(adjMatrix, j, vis);
        }
    }

    void DFSdemo(const V2<int> &adjMatrix)
    {
        V1<bool> vis(adjMatrix.size(), false);
        DFSdemo(adjMatrix, 1, vis);
    }
} // intro to graph

namespace Traversal
{
    
}

int main()
{
    // 1 -> 2,6
    // 2 -> 1,3,4
    // 3 -> 2
    // 4 -> 2,5
    // 5 -> 4,7
    // 6 -> 1,7,8
    // 7 -> 6,5
    // 8 -> 6
    V2<int> graph({{},
                   {2, 6},
                   {1, 3, 4},
                   {2},
                   {2, 5},
                   {4, 7},
                   {1, 7, 8},
                   {6, 5},
                   {6}});

    std::cout << "Hello, World!\n\n\n";
    std::cout << "BFS\n";
    IntroToGraph::BFSdemo(graph);
    std::cout << "\nDFS\n";
    IntroToGraph::DFSdemo(Utility::adjListToMatrix(graph));
};
