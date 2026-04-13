#include <cassert>
#include <iostream>
#include <queue>
#include <random>
#include <vector>
using namespace std;

class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> g(numCourses);
        vector<int> indeg(numCourses, 0);
        for (auto& e : prerequisites) {
            int a = e[0], b = e[1];
            g[b].push_back(a);
            indeg[a]++;
        }
        queue<int> q;
        for (int i = 0; i < numCourses; ++i) if (indeg[i] == 0) q.push(i); //确认起点位置
        int visited = 0;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            visited++;
            for (int v : g[u]) {
                if (--indeg[v] == 0) q.push(v);
            }
        }
        return visited == numCourses;
    }
};

static bool canFinishDFS(int n, const vector<vector<int>>& edges) {
    vector<vector<int>> g(n);
    for (auto& e : edges) g[e[1]].push_back(e[0]);
    vector<char> color(n, 0);
    function<bool(int)> dfs = [&](int u) -> bool {
        color[u] = 1;
        for (int v : g[u]) {
            if (color[v] == 1) return false;
            if (color[v] == 0 && !dfs(v)) return false;
        }
        color[u] = 2;
        return true;
    };
    for (int i = 0; i < n; ++i) if (color[i] == 0) if (!dfs(i)) return false;
    return true;
}

static vector<vector<int>> genDAG(mt19937& rng, int n, double p) {
    uniform_real_distribution<double> prob(0.0, 1.0);
    vector<vector<int>> edges;
    for (int u = 0; u < n; ++u) {
        for (int v = u + 1; v < n; ++v) {
            if (prob(rng) < p) edges.push_back({v, u});
        }
    }
    return edges;
}

static vector<vector<int>> addCycle(mt19937& rng, int n, vector<vector<int>> edges) {
    if (n < 2) return edges;
    uniform_int_distribution<int> pick(0, n - 1);
    int a = pick(rng), b = pick(rng);
    while (a == b) b = pick(rng);
    edges.push_back({a, b});
    edges.push_back({b, a});
    return edges;
}

int main() {
    Solution sol;
    {
        int n = 2;
        vector<vector<int>> pre{{1, 0}};
        assert(sol.canFinish(n, pre) == true);
    }
    {
        int n = 2;
        vector<vector<int>> pre{{1, 0}, {0, 1}};
        assert(sol.canFinish(n, pre) == false);
    }
    {
        int n = 4;
        vector<vector<int>> pre{{1, 0}, {2, 0}, {3, 1}, {3, 2}};
        assert(sol.canFinish(n, pre) == true);
    }
    {
        int n = 1;
        vector<vector<int>> pre{};
        assert(sol.canFinish(n, pre) == true);
    }

    mt19937 rng(20260410);
    for (int tc = 0; tc < 500; ++tc) {
        int n = uniform_int_distribution<int>(1, 200)(rng);
        double p = uniform_real_distribution<double>(0.01, 0.1)(rng);
        auto edges = genDAG(rng, n, p);
        vector<vector<int>> edgesCopy = edges;
        assert(sol.canFinish(n, edgesCopy) == canFinishDFS(n, edges));
        if (uniform_int_distribution<int>(0, 1)(rng) == 1) {
            auto edges2 = addCycle(rng, n, edges);
            vector<vector<int>> edges2Copy = edges2;
            assert(sol.canFinish(n, edges2Copy) == canFinishDFS(n, edges2));
        }
    }

    cout << "All tests passed." << "\n";
    return 0;
}

