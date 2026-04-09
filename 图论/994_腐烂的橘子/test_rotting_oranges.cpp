#include <cassert>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        int m = (int)grid.size();
        int n = (int)grid[0].size();
        queue<pair<int, int>> q;
        int fresh = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 2) q.push({i, j});
                else if (grid[i][j] == 1) fresh++;
            }
        }
        if (fresh == 0) return 0;
        if (q.empty()) return -1;
        int minutes = 0;
        const int di[4] = {1, -1, 0, 0};
        const int dj[4] = {0, 0, 1, -1};
        while (!q.empty()) {
            int sz = (int)q.size();
            for (int t = 0; t < sz; ++t) {
                auto [i, j] = q.front();
                q.pop();
                for (int k = 0; k < 4; ++k) {
                    int ni = i + di[k];
                    int nj = j + dj[k];
                    if (ni < 0 || ni >= m || nj < 0 || nj >= n) continue;
                    if (grid[ni][nj] != 1) continue;
                    grid[ni][nj] = 2;
                    fresh--;
                    q.push({ni, nj});
                }
            }
            if (!q.empty()) minutes++;
        }
        return fresh == 0 ? minutes : -1;
    }
};

static int bruteOrangesRotting(const vector<vector<int>>& g0) {
    int m = (int)g0.size();
    int n = (int)g0[0].size();
    vector<vector<int>> dist(m, vector<int>(n, -1));
    queue<pair<int, int>> q;
    int fresh = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (g0[i][j] == 2) {
                dist[i][j] = 0;
                q.push({i, j});
            } else if (g0[i][j] == 1) {
                fresh++;
            }
        }
    }
    if (fresh == 0) return 0;
    if (q.empty()) return -1;
    const int di[4] = {1, -1, 0, 0};
    const int dj[4] = {0, 0, 1, -1};
    int best = 0;
    while (!q.empty()) {
        auto [i, j] = q.front();
        q.pop();
        for (int k = 0; k < 4; ++k) {
            int ni = i + di[k];
            int nj = j + dj[k];
            if (ni < 0 || ni >= m || nj < 0 || nj >= n) continue;
            if (g0[ni][nj] == 0) continue;
            if (dist[ni][nj] != -1) continue;
            dist[ni][nj] = dist[i][j] + 1;
            q.push({ni, nj});
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (g0[i][j] == 1) {
                if (dist[i][j] == -1) return -1;
                best = max(best, dist[i][j]);
            }
        }
    }
    return best;
}

int main() {
    Solution sol;

    {
        vector<vector<int>> grid{
            {2, 1, 1},
            {1, 1, 0},
            {0, 1, 1}
        };
        auto copy = grid;
        assert(sol.orangesRotting(copy) == 4);
    }
    {
        vector<vector<int>> grid{
            {2, 1, 1},
            {0, 1, 1},
            {1, 0, 1}
        };
        auto copy = grid;
        assert(sol.orangesRotting(copy) == -1);
    }
    {
        vector<vector<int>> grid{{0}};
        auto copy = grid;
        assert(sol.orangesRotting(copy) == 0);
    }
    {
        vector<vector<int>> grid{{2}};
        auto copy = grid;
        assert(sol.orangesRotting(copy) == 0);
    }
    {
        vector<vector<int>> grid{{1}};
        auto copy = grid;
        assert(sol.orangesRotting(copy) == -1);
    }

    mt19937 rng(20260409);
    uniform_int_distribution<int> mDist(1, 12);
    uniform_int_distribution<int> nDist(1, 12);
    uniform_int_distribution<int> cellDist(0, 2);

    for (int tc = 0; tc < 3000; ++tc) {
        int m = mDist(rng);
        int n = nDist(rng);
        vector<vector<int>> grid(m, vector<int>(n, 0));
        bool hasFresh = false;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                grid[i][j] = cellDist(rng);
                if (grid[i][j] == 1) hasFresh = true;
            }
        }
        auto copy = grid;
        int got = sol.orangesRotting(copy);
        int want = bruteOrangesRotting(grid);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

