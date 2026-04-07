#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = (int)matrix.size();
        int n = (int)matrix[0].size();
        int lo = 0;
        int hi = m * n - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            int r = mid / n;
            int c = mid % n;
            int v = matrix[r][c];
            if (v == target) return true;
            if (v < target) lo = mid + 1;
            else hi = mid - 1;
        }
        return false;
    }
};

static bool bruteSearch(const vector<vector<int>>& a, int target) {
    for (const auto& row : a) {
        for (int x : row) {
            if (x == target) return true;
        }
    }
    return false;
}

static vector<vector<int>> genMatrix(mt19937& rng, int m, int n) {
    uniform_int_distribution<int> startDist(-10000, 10000);
    uniform_int_distribution<int> stepDist(0, 3);
    int cur = startDist(rng);
    vector<vector<int>> a(m, vector<int>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            cur += 1 + stepDist(rng);
            a[i][j] = cur;
        }
    }
    return a;
}

int main() {
    Solution sol;

    {
        vector<vector<int>> matrix{{1, 3, 5, 7}, {10, 11, 16, 20}, {23, 30, 34, 60}};
        assert(sol.searchMatrix(matrix, 3) == true);
        assert(sol.searchMatrix(matrix, 13) == false);
    }
    {
        vector<vector<int>> matrix{{-5}};
        assert(sol.searchMatrix(matrix, -5) == true);
        assert(sol.searchMatrix(matrix, 0) == false);
    }
    {
        vector<vector<int>> matrix{{1, 2, 3, 4, 5}};
        assert(sol.searchMatrix(matrix, 1) == true);
        assert(sol.searchMatrix(matrix, 5) == true);
        assert(sol.searchMatrix(matrix, 6) == false);
    }
    {
        vector<vector<int>> matrix{{1}, {3}, {5}};
        assert(sol.searchMatrix(matrix, 3) == true);
        assert(sol.searchMatrix(matrix, 4) == false);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> mDist(1, 100);
    uniform_int_distribution<int> nDist(1, 100);
    uniform_int_distribution<int> queryDist(-20000, 20000);

    for (int tc = 0; tc < 2000; ++tc) {
        int m = mDist(rng);
        int n = nDist(rng);
        vector<vector<int>> a = genMatrix(rng, m, n);
        vector<vector<int>> copy = a;
        for (int q = 0; q < 50; ++q) {
            int target = queryDist(rng);
            bool got = sol.searchMatrix(copy, target);
            bool want = bruteSearch(a, target);
            assert(got == want);
        }
    }

    cout << "All tests passed." << "\n";
    return 0;
}

