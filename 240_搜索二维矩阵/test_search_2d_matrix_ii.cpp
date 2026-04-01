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
        int i = 0;
        int j = n - 1;
        while (i < m && j >= 0) {
            int v = matrix[i][j];
            if (v == target) return true;
            if (v > target) {
                j--;
            } else {
                i++;
            }
        }
        return false;
    }
};

static bool bruteSearch(const vector<vector<int>>& matrix, int target) {
    for (const auto& row : matrix) {
        for (int x : row) {
            if (x == target) return true;
        }
    }
    return false;
}

int main() {
    Solution sol;

    {
        vector<vector<int>> matrix{
            {1, 4, 7, 11, 15},
            {2, 5, 8, 12, 19},
            {3, 6, 9, 16, 22},
            {10, 13, 14, 17, 24},
            {18, 21, 23, 26, 30},
        };
        assert(sol.searchMatrix(matrix, 5) == true);
        assert(sol.searchMatrix(matrix, 20) == false);
    }
    {
        vector<vector<int>> matrix{{-5}};
        assert(sol.searchMatrix(matrix, -5) == true);
        assert(sol.searchMatrix(matrix, 0) == false);
    }
    {
        vector<vector<int>> matrix{{1, 2, 3, 4}};
        assert(sol.searchMatrix(matrix, 3) == true);
        assert(sol.searchMatrix(matrix, 5) == false);
    }
    {
        vector<vector<int>> matrix{{1}, {3}, {5}};
        assert(sol.searchMatrix(matrix, 3) == true);
        assert(sol.searchMatrix(matrix, 2) == false);
    }

    mt19937 rng(20260330);
    uniform_int_distribution<int> mDist(1, 30);
    uniform_int_distribution<int> nDist(1, 30);
    uniform_int_distribution<int> stepDist(0, 3);
    uniform_int_distribution<int> startDist(-10, 10);
    uniform_int_distribution<int> queryDist(-30, 80);

    for (int tc = 0; tc < 3000; ++tc) {
        int m = mDist(rng);
        int n = nDist(rng);
        int base = startDist(rng);
        vector<vector<int>> matrix(m, vector<int>(n));
        int cur = base;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                cur += stepDist(rng);
                matrix[i][j] = cur;
            }
            cur += stepDist(rng);
        }

        for (int q = 0; q < 50; ++q) {
            int target = queryDist(rng);
            vector<vector<int>> copy = matrix;
            bool got = sol.searchMatrix(copy, target);
            bool want = bruteSearch(matrix, target);
            assert(got == want);
        }
    }

    cout << "All tests passed." << "\n";
    return 0;
}

