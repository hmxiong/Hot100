#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int m = (int)grid.size();
        int n = (int)grid[0].size();
        vector<int> dp(n, 0);
        dp[0] = grid[0][0];
        for (int j = 1; j < n; ++j) dp[j] = dp[j - 1] + grid[0][j];
        for (int i = 1; i < m; ++i) {
            dp[0] += grid[i][0]; // 注意这里要累加，而不是赋值
            for (int j = 1; j < n; ++j) dp[j] = min(dp[j], dp[j - 1]) + grid[i][j];
        }
        return dp[n - 1];
    }
};

static int bruteMinPathSum(const vector<vector<int>>& g) {
    int m = (int)g.size();
    int n = (int)g[0].size();
    vector<vector<int>> dp(m, vector<int>(n, 0));
    dp[0][0] = g[0][0];
    for (int j = 1; j < n; ++j) dp[0][j] = dp[0][j - 1] + g[0][j];
    for (int i = 1; i < m; ++i) dp[i][0] = dp[i - 1][0] + g[i][0];
    for (int i = 1; i < m; ++i) {
        for (int j = 1; j < n; ++j) dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + g[i][j];
    }
    return dp[m - 1][n - 1];
}

int main() {
    Solution sol;

    {
        vector<vector<int>> grid{{1,3,1},{1,5,1},{4,2,1}};
        auto copy = grid;
        assert(sol.minPathSum(copy) == 7);
    }
    {
        vector<vector<int>> grid{{1,2,3},{4,5,6}};
        auto copy = grid;
        assert(sol.minPathSum(copy) == 12);
    }
    {
        vector<vector<int>> grid{{5}};
        auto copy = grid;
        assert(sol.minPathSum(copy) == 5);
    }
    {
        vector<vector<int>> grid{{1, 2, 3, 4}};
        auto copy = grid;
        assert(sol.minPathSum(copy) == 10);
    }
    {
        vector<vector<int>> grid{{1},{2},{3},{4}};
        auto copy = grid;
        assert(sol.minPathSum(copy) == 10);
    }

    mt19937 rng(20260409);
    uniform_int_distribution<int> mDist(1, 50);
    uniform_int_distribution<int> nDist(1, 50);
    uniform_int_distribution<int> valDist(0, 20);
    for (int tc = 0; tc < 2000; ++tc) {
        int m = mDist(rng);
        int n = nDist(rng);
        vector<vector<int>> g(m, vector<int>(n, 0));
        for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) g[i][j] = valDist(rng);
        auto copy = g;
        int got = sol.minPathSum(copy);
        int want = bruteMinPathSum(g);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

