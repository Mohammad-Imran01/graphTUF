#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <stack>
#include <utility>

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

class NumberOfProvinces
{
    V1<int> vis;
    int res;
    void dfsTraverse(const V2<int> &adjMat, int node)
    {
        if (vis[node])
            return;
        vis[node] = true;
        for (int curr = 0; curr < adjMat.size(); ++curr)
            if (adjMat[node][curr] && !vis[curr])
                dfsTraverse(adjMat, curr);
    }

public:
    NumberOfProvinces()
    {
        res = 0;
    }
    //: adj matrix
    int dfsSolution(const V2<int> &graph)
    {
        const int len = graph.size();
        vis = V1<int>(len, false);

        for (int i = 0; i < len; ++i)
        {
            if (vis[i])
                continue;
            dfsTraverse(graph, i);
            ++res;
        }
        return res;
    }
};

class RottenOranges
{

public:
    int orangesRotting(V2<int> &grid)
    {
        if (grid.empty())
            return 0;
        int m = grid.size();
        int n = grid[0].size();

        std::queue<std::pair<int, int>> q;

        int res = -1;
        int fresh = 0;

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (grid[i][j] == 2)
                    q.push({i, j});
                else if (grid[i][j] == 1)
                    ++fresh;
            }
        }

        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, -1, 0, 1};

        while (q.size())
        {
            int sz = q.size();
            while (sz--)
            {
                auto [row, col] = q.front();
                q.pop();
                for (int i = 0; i < 4; ++i)
                {
                    int r = row + dx[i];
                    int c = col + dy[i];

                    if ((r >= 0) && (c >= 0) && (r < m) && (c < n) && (grid[r][c] == 1))
                    {
                        q.push({r, c});
                        grid[r][c] = 2;
                        --fresh;
                    }
                }
            }
            ++res;
        }

        if (fresh > 0)
            return -1;
        return std::max(res, 0);
    }
};

class FloodFill
{
public:
    V2<int> solve(V2<int> &graph, int sr, int sc, int color)
    {
        int m = graph.size();
        if (graph.empty() || graph.at(sr).at(sc) == color)
            return graph;
        int n = graph.front().size();

        std::queue<std::pair<int, int>> q;
        q.push({sr, sc});

        const int target = graph.at(sr).at(sc);

        graph[sr][sc] = color;

        int dx[4]{0, 0, 1, -1};
        int dy[4]{-1, 1, 0, 0};

        while (q.size())
        {
            auto [x, y] = q.front();
            q.pop();
            graph.at(x).at(y) = color;

            for (int i = 0; i < 4; ++i)
            {
                int row = x + dx[i];
                int col = y + dy[i];

                if ((row >= 0) && (col >= 0) && (row < m) && (col < n) && graph.at(row).at(col) == target)
                {
                    graph.at(row).at(col) = color;
                    q.push({row, col});
                }
            }
        }
    }
};

class DetectCycle
{
    V1<int> vis;
    V2<int> adjList;
    bool dfs(int node)
    {
        if (vis[node] == 2) // visited return👍
            return false;
        if (vis[node] == 1) // visiting already🤷‍♀️
            return true;
        vis[node] = 1; // visiting👏

        for (const int &curr : adjList[node])
        {
            if (dfs(curr))
                return true;
        }
        vis[node] = 2;
        return false;
    }

public:
    bool canFinish(int len, V2<int> &pre)
    {
        if (pre.size() < 2)
            return true;
        vis = V1<int>(len, 0);
        adjList = V2<int>(len);

        for (auto &p : pre)
            adjList[p[0]].push_back(p[1]);

        for (int i = 0; i < len; ++i)
        {
            if (vis[i] == 0 && dfs(i))
                return false;
        }
        return true;
    }
};

// Nearest zero  for each cell, there is atleast one zero in the grid
// Solved using multi bfs
class UpdateMatrix
{
public:
    V2<int> solve(V2<int> &mat)
    {
        if (mat.empty())
            return mat;
        const int m = mat.size();
        const int n = mat[0].size();

        std::queue<std::pair<int, int>> q;

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (mat[i][j])
                    mat[i][j] = 1e8;
                else
                    q.push({i, j});
            }
        }

        int dx[]{0, 0, -1, 1};
        int dy[]{-1, 1, 0, 0};
        while (!q.empty())
        {
            auto [x, y] = q.front();
            q.pop();

            for (int k = 0; k < 4; ++k)
            {
                int r = x + dx[k];
                int c = y + dy[k];

                if (r >= 0 && c >= 0 && r < m && c < n && mat[r][c] > mat[x][y] + 1)
                {
                    mat[r][c] = mat[x][y] + 1;
                    q.push({r, c});
                }
            }
        }

        return mat;
    }
};

class SurroundedRegions
{
    int m, n;

    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};
    void solve(V2<char> &board, int x, int y)
    {
        if (x < 0 || y < 0 || x >= m || y >= n || board[x][y] != 'O')
            return;
        board[x][y] = '9';
        for (int i = 0; i < 4; ++i)
            solve(board, x + dx[i], y + dy[i]);
    }

public:
    void solve(V2<char> &board)
    {
        if (board.empty())
            return;                            //
        m = board.size(), n = board[0].size(); //

        for (int i = 0; i < m; ++i)
        {
            solve(board, i, 0);
            solve(board, i, n - 1);
        }
        for (int j = 0; j < n; ++j)
        {
            solve(board, 0, j);
            solve(board, m - 1, j);
        }

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                if (board[i][j] == 'O')
                    board[i][j] = 'X';
                else if (board[i][j] == '9')
                    board[i][j] = 'O';
            }
        }
    }
};

class NumEnclaves
{
    int m, n;
    void dfs(V2<int> &grid, int x, int y)
    {
        if (x < 0 || y < 0 || x >= grid.size() || y >= grid[x].size() || grid[x][y] < 1)
            return;
        grid[x][y] = -1;

        std::cout << "\n"
                  << x << ", " << y << ", " << grid.size() << ", " << grid[x].size();

        dfs(grid, x, y - 1);
        dfs(grid, x, y + 1);
        dfs(grid, x - 1, y);
        dfs(grid, x + 1, y);
    }

public:
    int numEnclaves(V2<int> &grid)
    {
        if (grid.empty())
            return 0;
        m = grid.size();
        n = grid[0].size();

        // Remove boundary-connected land
        for (int i = 0; i < m; ++i)
        {
            dfs(grid, i, 0);
            dfs(grid, i, n - 1);
        }
        for (int i = 0; i < n; ++i)
        {
            dfs(grid, 0, i);
            dfs(grid, m - 1, i);
        }

        // Count remaining land
        int res = 0;
        for (auto &g : grid)
        {
            for (int num : g)
            {
                if (num == 1)
                    ++res;
            }
        }

        return res;
    }
};

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
