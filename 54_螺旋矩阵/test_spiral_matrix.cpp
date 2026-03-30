#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m = (int)matrix.size();
        int n = (int)matrix[0].size();
        vector<int> res;
        res.reserve((size_t)m * (size_t)n);
        int top = 0, bottom = m - 1, left = 0, right = n - 1;
        while (top <= bottom && left <= right) {
            for (int j = left; j <= right; ++j) res.push_back(matrix[top][j]);
            top++;
            for (int i = top; i <= bottom; ++i) res.push_back(matrix[i][right]);
            right--;
            if (top <= bottom) {
                for (int j = right; j >= left; --j) res.push_back(matrix[bottom][j]);
                bottom--;
            }
            if (left <= right) {
                for (int i = bottom; i >= top; --i) res.push_back(matrix[i][left]);
                left++;
            }
        }
        return res;
    }
};

static vector<int> spiralOrderRef(const vector<vector<int>>& a) {
    int m = (int)a.size();
    int n = (int)a[0].size();
    vector<int> res;
    res.reserve((size_t)m * (size_t)n);
    vector<vector<char>> vis(m, vector<char>(n, 0));
    int dirs[4][2] = {{0,1},{1,0},{0,-1},{-1,0}};
    int d = 0, i = 0, j = 0;
    for (int k = 0; k < m*n; ++k) {
        res.push_back(a[i][j]);
        vis[i][j] = 1;
        int ni = i + dirs[d][0];
        int nj = j + dirs[d][1];
        if (ni < 0 || ni >= m || nj < 0 || nj >= n || vis[ni][nj]) {
            d = (d + 1) % 4;
            ni = i + dirs[d][0];
            nj = j + dirs[d][1];
        }
        i = ni; j = nj;
    }
    return res;
}

static void expectEq(const vector<int>& a, const vector<int>& b) {
    assert(a == b);
}

int main() {
    Solution sol;

    {
        vector<vector<int>> m{{1,2,3},{4,5,6},{7,8,9}};
        auto got = sol.spiralOrder(m);
        expectEq(got, vector<int>({1,2,3,6,9,8,7,4,5}));
    }
    {
        vector<vector<int>> m{{1,2,3,4},{5,6,7,8},{9,10,11,12}};
        auto got = sol.spiralOrder(m);
        expectEq(got, vector<int>({1,2,3,4,8,12,11,10,9,5,6,7}));
    }
    {
        vector<vector<int>> m{{1}};
        auto got = sol.spiralOrder(m);
        expectEq(got, vector<int>({1}));
    }
    {
        vector<vector<int>> m{{1,2},{3,4}};
        auto got = sol.spiralOrder(m);
        expectEq(got, vector<int>({1,2,4,3}));
    }

    mt19937 rng(20260330);
    uniform_int_distribution<int> mDist(1, 10);
    uniform_int_distribution<int> nDist(1, 10);
    int val = 1;
    for (int tc = 0; tc < 2000; ++tc) {
        int m = mDist(rng);
        int n = nDist(rng);
        vector<vector<int>> a(m, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) a[i][j] = val++;
        }
        auto got = sol.spiralOrder(a);
        auto want = spiralOrderRef(a);
        expectEq(got, want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

