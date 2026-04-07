#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = (int)matrix.size();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
        for (int i = 0; i < n; ++i) {
            reverse(matrix[i].begin(), matrix[i].end());
        }
    }
};

static vector<vector<int>> rotateBrute(const vector<vector<int>>& a) {
    int n = (int)a.size();
    vector<vector<int>> b(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b[j][n - 1 - i] = a[i][j];
        }
    }
    return b;
}

static void expectEq(const vector<vector<int>>& a, const vector<vector<int>>& b) {
    assert(a == b);
}

int main() {
    Solution sol;

    {
        vector<vector<int>> m{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        sol.rotate(m);
        expectEq(m, vector<vector<int>>({{7, 4, 1}, {8, 5, 2}, {9, 6, 3}}));
    }
    {
        vector<vector<int>> m{{5, 1, 9, 11}, {2, 4, 8, 10}, {13, 3, 6, 7}, {15, 14, 12, 16}};
        sol.rotate(m);
        expectEq(m, vector<vector<int>>({{15, 13, 2, 5}, {14, 3, 4, 1}, {12, 6, 8, 9}, {16, 7, 10, 11}}));
    }
    {
        vector<vector<int>> m{{42}};
        sol.rotate(m);
        expectEq(m, vector<vector<int>>({{42}}));
    }

    mt19937 rng(20260328);
    uniform_int_distribution<int> nDist(1, 20);
    uniform_int_distribution<int> valDist(-5, 5);

    for (int t = 0; t < 2000; ++t) {
        int n = nDist(rng);
        vector<vector<int>> m(n, vector<int>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) m[i][j] = valDist(rng);
        }

        auto expected = rotateBrute(m);
        sol.rotate(m);
        expectEq(m, expected);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

