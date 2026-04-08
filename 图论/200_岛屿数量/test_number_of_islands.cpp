#include <cassert>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int m = (int)grid.size();
        int n = (int)grid[0].size();
        int cnt = 0;

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] != '1') continue;
                cnt++;
                bfs(grid, i, j);
            }
        }
        return cnt;
    }

private: // 在这里bfs的主要作用是将访问过的节点标记为'0'，以及将相邻的'1'也标记为'0'，以及将相邻的'0'标记为'1'也标记为'0'
    static void bfs(vector<vector<char>>& grid, int si, int sj) {
        int m = (int)grid.size();
        int n = (int)grid[0].size();
        queue<pair<int, int>> q;
        q.push({si, sj});
        grid[si][sj] = '0';
        const int di[4] = {1, -1, 0, 0}; // 作用是上下左右四个方向的偏移量
        const int dj[4] = {0, 0, 1, -1}; // 作用是上下左右四个方向的偏移量

        while (!q.empty()) {
            auto [i, j] = q.front();
            q.pop();
            for (int k = 0; k < 4; ++k) {
                int ni = i + di[k];
                int nj = j + dj[k];
                if (ni < 0 || ni >= m || nj < 0 || nj >= n) continue;
                if (grid[ni][nj] != '1') continue;
                grid[ni][nj] = '0';
                q.push({ni, nj});
            }
        }
    }

    
};

static int bruteNumIslands(vector<vector<char>> grid) {
    int m = (int)grid.size();
    int n = (int)grid[0].size();
    int cnt = 0;
    queue<pair<int, int>> q;
    const int di[4] = {1, -1, 0, 0};
    const int dj[4] = {0, 0, 1, -1};

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] != '1') continue;
            cnt++;
            grid[i][j] = '0';
            q.push({i, j});
            while (!q.empty()) {
                auto [x, y] = q.front();
                q.pop();
                for (int k = 0; k < 4; ++k) {
                    int nx = x + di[k];
                    int ny = y + dj[k];
                    if (nx < 0 || nx >= m || ny < 0 || ny >= n) continue;
                    if (grid[nx][ny] != '1') continue;
                    grid[nx][ny] = '0';
                    q.push({nx, ny});
                }
            }
        }
    }
    return cnt;
}

static vector<vector<char>> genGrid(mt19937& rng, int m, int n, double landProb) {
    uniform_real_distribution<double> dist(0.0, 1.0);
    vector<vector<char>> g(m, vector<char>(n, '0'));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            g[i][j] = (dist(rng) < landProb) ? '1' : '0';
        }
    }
    return g;
}

int main() {
    Solution sol;

    {
        vector<vector<char>> grid{
            {'1', '1', '1', '1', '0'},
            {'1', '1', '0', '1', '0'},
            {'1', '1', '0', '0', '0'},
            {'0', '0', '0', '0', '0'},
        };
        vector<vector<char>> copy = grid;
        assert(sol.numIslands(copy) == 1);
    }
    {
        vector<vector<char>> grid{
            {'1', '1', '0', '0', '0'},
            {'1', '1', '0', '0', '0'},
            {'0', '0', '1', '0', '0'},
            {'0', '0', '0', '1', '1'},
        };
        vector<vector<char>> copy = grid;
        assert(sol.numIslands(copy) == 3);
    }
    {
        vector<vector<char>> grid{{'0'}};
        vector<vector<char>> copy = grid;
        assert(sol.numIslands(copy) == 0);
    }
    {
        vector<vector<char>> grid{{'1'}};
        vector<vector<char>> copy = grid;
        assert(sol.numIslands(copy) == 1);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> mDist(1, 60);
    uniform_int_distribution<int> nDist(1, 60);
    uniform_real_distribution<double> pDist(0.05, 0.95);

    for (int tc = 0; tc < 500; ++tc) {
        int m = mDist(rng);
        int n = nDist(rng);
        double p = pDist(rng);
        vector<vector<char>> grid = genGrid(rng, m, n, p);
        vector<vector<char>> copy = grid;
        int got = sol.numIslands(copy);
        int want = bruteNumIslands(grid);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

