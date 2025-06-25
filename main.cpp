#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <stack>
#include <utility>
#include <functional>
#include <unordered_map>
#include <unordered_set>

template <typename Type>
using V1 = std::vector<Type>;
template <typename Type>
using V2 = std::vector<V1<Type>>;

template <typename TYPE1, typename TYPE2>
using Pr = std::pair<TYPE1, TYPE2>;

template <typename t1, typename t2>
using V2Pair = V2<std::pair<t1, t2>>;

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

class WordLadder
{

    bool almostEqual(const std::string &s1, const std::string &s2)
    {
        if (s1.size() != s2.size())
            return false;
        int count = 0;

        for (int i = 0; i < s1.size(); ++i)
        {
            count += (s1[i] != s2[i]);
            if (count > 1)
                return false;
        }
        return count == 1;
    }

public:
    int ladderLength(std::string beginWord, std::string endWord, V1<std::string> &wordList)
    {
        std::unordered_set<std::string> words(wordList.begin(), wordList.end());
        std::unordered_set<std::string> vis;
        std::queue<std::string> q;
        q.push(beginWord);
        vis.insert(beginWord);

        int time = 1;

        while (q.size())
        {
            int len = q.size();
            while (len--)
            {
                std::string word = q.front();
                q.pop();
                if (word == endWord)
                    return time;
                for (auto next : words)
                {
                    if (!vis.count(next) && almostEqual(word, next))
                    {
                        q.push(next);
                        vis.insert(next);
                    }
                }
            }
            ++time;
        }
        return 0;
    }
};

class Bipartite
{
    int *vis = nullptr;
    bool colorIt(int node, V2<int> &graph)
    {
        std::queue<int> q;
        q.push(node);
        vis[node] = 0;
        while (q.size())
        {
            int curr = q.front();
            q.pop();
            for (int it : graph[curr])
            {
                if (vis[it] == vis[curr])
                    return false;
                if (vis[it] == -1)
                {
                    vis[it] = !vis[curr];
                    q.push(it);
                }
            }
        }
        return true;
    }

public:
    bool isBipartite(V2<int> &graph)
    {
        if (graph.empty())
            return true;
        const int len = graph.size();
        vis = new int[len];
        std::fill(vis, vis + len, -1);

        for (int i = 0; i < len; ++i)
        {
            if (vis[i] == -1)
                if (!colorIt(i, graph))
                    return false;
        }
        return true;
    }
};

class CanTakeourses
{
    V1<int> *graph = nullptr;
    V1<int> res;
    bool loop(int node, V1<int> &vis, V2<int> &adj)
    {
        if (vis[node] == 1)
            return true;
        if (vis[node] == 2)
            return false;
        vis[node] = 1;

        for (auto curr : adj[node])
            if (loop(curr, vis, adj))
                return true;

        res.push_back(node);
        vis[node] = 2;

        return false;
    }

public:
    V1<int> findOrder(int len, V2<int> &graph_xy)
    {
        // if(len < 2) return true;
        V2<int> adj(len, V1<int>());

        for (auto arr : graph_xy)
            adj[arr[0]].push_back(arr[1]);

        V1<int> vis(len, 0);

        for (int i = 0; i < len; ++i)
        {
            if (!vis[i])
            {
                if (loop(i, vis, adj))
                {
                    return {};
                }
            }
        }

        return res;
        // return {res.rbegin(), res.rend()};
    }
};

class ShortestPathUnitDistantEdges
{
public:
    std::vector<int> solve(const std::vector<std::vector<int>> &adj, int src)
    {
        const int len = adj.size();
        std::vector<int> vis(len, 1e8);
        std::queue<std::pair<int, int>> q;

        vis[src] = 0;
        q.push({src, 0});

        while (q.size())
        {
            int sz = q.size();

            while (sz--)
            {
                // int node = q.front().first;
                // int cost = q.front().second;
                auto [node, cost] = q.front();

                q.pop();

                for (int curr : adj[node])
                {
                    if (vis[curr] > cost + 1)
                    {
                        vis[curr] = cost + 1;
                        q.push({curr, cost + 1});
                    }
                }
            }
        }
        for (int i = 0; i < len; ++i)
            if (vis[i] >= 1e8)
                vis[i] = -1;

        return vis;
    }
};

class ShortestPathDAG
{
    std::stack<int> stk;
    void dfs(int src, const V2Pair<int, int> &adj, std::vector<bool> &vis)
    {
        vis[src] = true;
        for (auto currNode : adj[src])
            if (!vis[currNode.first])
                dfs(currNode.first, adj, vis);
        stk.push(src);
    }

public:
    V1<int> solve(int V, const V2<int> &edges, int src = 0)
    {
        if (V < 1)
            return V1<int>{};

        V2Pair<int, int> adj(V);

        for (auto edge : edges)
            adj[edge[0]].push_back({edge[1], edge[2]});

        V1<bool> vis(V, false);
        for (int i = 0; i < V; ++i)
            if (!vis[i])
                dfs(i, adj, vis);

        V1<int> res(V, 1e9);
        res[src] = 0;

        while (stk.size())
        {
            int src = stk.top();
            stk.pop();

            if (res[src] < 1e9)
            {
                for (auto edge : adj[src])
                {
                    if (res[edge.first] > edge.second + res[src])
                        res[edge.first] = edge.second + res[src];
                }
            }
        }

        for (int i = 0; i < V; ++i)
            if (res[i] >= 1e9)
                res[i] = -1;
        return res;
    }
};

class ShortestPathInMaze
{
    V1<int> dx = {-1, -1, -1, 0, 0, 1, 1, 1};
    V1<int> dy = {-1, 0, 1, -1, 1, -1, 0, 1};

public:
    int solve(V2<int> &mat)
    {
        if (mat.empty() || mat.front().empty())
            return -1;
        if (mat[0][0] || mat.back().back())
            return -1;

        int m = mat.size();
        int n = mat.front().size();

        for (auto &rows : mat)
            for (int &num : rows)
                num = num ? -1 : 1e8;

        mat[0][0] = 1;
        std::queue<Pr<int, int>> q;
        q.push({0, 0});

        while (q.size())
        {
            int len = q.size();
            while (len--)
            {
                auto [row, col] = q.front();
                q.pop();

                for (int i = 0; i < 8; ++i)
                {
                    int r = row + dx[i];
                    int c = col + dy[i];

                    if (r < 0 || c < 0 || r >= m || c >= n || mat[r][c] < 0)
                        continue;
                    if (mat[r][c] > mat[row][col] + 1)
                    {
                        mat[r][c] = mat[row][col] + 1;
                        q.push({r, c});
                    }
                }
            }
        }

        return (mat.back().back() < 0 || mat.back().back() >= 1e8)
                   ? -1
                   : mat.back().back();
    }
};

class PathMinEffort
{
public:
    int solve(const V2<int> mat)
    {
        if (mat.empty() || mat.front().empty())
            return -1;

        int
            m = mat.size(),
            n = mat[0].size();

        std::priority_queue<
            Pr<int, Pr<int, int>>,
            V1<Pr<int, Pr<int, int>>>,
            std::greater<>>
            pq;

        V2<int> memo(m, V1<int>(n, 1e8));

        memo[0][0] = 0;
        pq.push({0, {0, 0}});

        int dx[4] = {-1, 0, 0, 1};
        int dy[4] = {0, -1, 1, 0};

        while (pq.size())
        {
            auto cost = pq.top().first;
            auto [row, col] = pq.top().second;
            pq.pop();

            if (row == m - 1 && col == n - 1)
                return cost;

            for (int i = 0; i < 4; ++i)
            {
                int r = dx[i] + row;
                int c = dy[i] + col;

                if (r < 0 || c < 0 || r >= m || c >= n)
                    continue;

                int diff = std::abs(mat[r][c] - mat[row][col]);
                int newDiff = std::max(diff, cost);

                if (memo[r][c] > newDiff)
                {
                    memo[r][c] = newDiff;
                    pq.push({newDiff, {r, c}});
                }
            }
        }
        return -1;
    }
};

class CheapestFlightsWithKStops
{
public:
    int solve(int n, int k, int src, int dest, V2<int> flights)
    {
        V2<Pr<int, int>> adj(n);
        for (const auto &flight : flights)
        {
            adj[flight[0]].push_back({flight[1], flight[2]});
        }
        std::queue<V1<int>> q;
        V1<int> vis(n, 1e9);
        vis[src] = 0;
        q.push({0, src, 0});
        // {K, src, Cost}

        while (q.size())
        {
            auto it = q.front();
            q.pop();
            int stops = it[0];
            int node = it[1];
            int cost = it[2];

            if (stops > k)
                continue;
            for (auto curr : adj[node])
            {
                int currNode = curr.first;
                int currCost = curr.second;

                if (cost + currCost < vis[currNode])
                {
                    vis[currNode] = cost + currCost;
                    q.push({stops + 1, currNode, vis[currNode]});
                }
            }
        }
        return (vis[dest] >= 1e9) ? -1 : vis[dest];
    }
};

class NetworkDelayTime
{
public:
    int solve(V2<int> &times, int n, int k)
    {
        V2<Pr<int, int>> adj(n + 1);
        for (auto &time : times)
            adj[time[0]].push_back({time[1], time[2]});
        V1<int> vis(n + 1, 1e9);
        vis[k] = 0;
        std::queue<V1<int>> q;
        q.push({k, 0});

        while (q.size())
        {
            int src = q.front()[0], cost = q.front()[1];
            q.pop();

            for (auto [next, costNext] : adj[src])
            {
                if (cost + costNext < vis[next])
                {
                    vis[next] = cost + costNext;
                    q.push({next, cost + costNext});
                }
            }
        }
        int res = -1e9;
        for (int i = 1; i <= n; ++i)
        {
            if (vis[i] >= 1e9)
                return -1;
            res = std::max(res, vis[i]);
        }
        return res;
    }
};

class NumberOfMinCostWaysToDest
{
    const long long MOD = 1e9 + 7;

public:
    int solve(int n, const V2<int> &edges)
    {
        V2<Pr<int, int>> adj(n);
        for (const auto &edge : edges)
        {
            adj[edge[0]].push_back({edge[1], edge[2]});
            adj[edge[1]].push_back({edge[0], edge[2]});
        }
        V1<int> dis(n, LLONG_MAX), ways(n, 0);

        dis[0] = 0;
        ways[0] = 1;

        std::priority_queue<Pr<int, int>, V1<Pr<int, int>>, std::greater<>> pq;

        while (pq.size())
        {
            auto [cost, node] = pq.top();
            pq.pop();

            if (cost > dis[node])
                continue;

            for (const auto &[curr, wt] : adj[node])
            {
                long long newCost = wt + cost;
                if (newCost < dis[curr])
                {
                    dis[curr] = newCost;
                    ways[curr] = ways[node];
                    pq.push({newCost, curr});
                }
                else if (newCost == dis[curr])
                {
                    ways[curr] = (ways[curr] + ways[node]) % MOD;
                }
            }
        }
        return ways.back();
    }
};

// User function Template for C++

class MinimumMultiplicationsToReachEnd
{
public:
    int minimumMultiplications(V1<int> &nums, int start, int end)
    {
        std::queue<Pr<int, int>> q;
        V1<int> vis(100000, 1e9);

        q.push({0, start});
        vis[start] = 0;
        // wt, node

        while (q.size())
        {
            int steps = q.front().first, node = q.front().second;
            q.pop();

            if (node == end)
                return steps;

            for (int num : nums)
            {
                int newStart = (node * num) % 100000;
                if (vis[newStart] > steps + 1)
                {
                    vis[newStart] = steps + 1;
                    q.push({steps + 1, newStart});
                }
            }
        }

        return -1;
    }
};

// User function Template for C++

class BellManShortestPath
{
public:
    V1<int> bellmanFord(int V, V2<int> &edges, int src)
    {
        // Code here
        V1<int> dis(V, 1e8);
        dis[src] = 0;

        bool negCycle = false;

        for (int iter = 0; iter < V - 1; ++iter)
        {
            for (const auto &edge : edges)
            {
                if (dis[edge[0]] != 1e8)
                    if (dis[edge[0]] + edge[2] < dis[edge[1]])
                        dis[edge[1]] = dis[edge[0]] + edge[2];
            }
        }
        for (const auto &edge : edges)
        {
            if (dis[edge[0]] != 1e8)
                if (dis[edge[0]] + edge[2] < dis[edge[1]])
                {
                    negCycle = true;
                    break;
                }
        }
        if (negCycle)
            return {-1};
        return dis;
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
