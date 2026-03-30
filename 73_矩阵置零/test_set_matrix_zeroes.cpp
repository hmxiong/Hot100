#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = (int)matrix.size();
        int n = (int)matrix[0].size();

        bool firstRowZero = false;
        bool firstColZero = false;
        
        //作用：记录第一行是否有0
        //作用：记录第一列是否有0
        for (int j = 0; j < n; ++j) {
            if (matrix[0][j] == 0) {
                firstRowZero = true;
                break;
            }
        }
        for (int i = 0; i < m; ++i) {
            if (matrix[i][0] == 0) {
                firstColZero = true;
                break;
            }
        }

        //作用：将第一行的0，将第一列的0，转换为0
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        //寻找第一行是否有0，然后将第一行的0，转换为0
        for (int i = 1; i < m; ++i) {
            if (matrix[i][0] == 0) {
                for (int j = 1; j < n; ++j) matrix[i][j] = 0;
            }
        }
        for (int j = 1; j < n; ++j) {
            if (matrix[0][j] == 0) {
                for (int i = 1; i < m; ++i) matrix[i][j] = 0;
            }
        }

        if (firstRowZero) {
            for (int j = 0; j < n; ++j) matrix[0][j] = 0;
        }
        if (firstColZero) {
            for (int i = 0; i < m; ++i) matrix[i][0] = 0;
        }
    }
};

static vector<vector<int>> bruteSetZeroes(vector<vector<int>> a) {
    int m = (int)a.size();
    int n = (int)a[0].size();
    vector<int> row(m, 0), col(n, 0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (a[i][j] == 0) {
                row[i] = 1;
                col[j] = 1;
            }
        }
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (row[i] || col[j]) a[i][j] = 0;
        }
    }
    return a;
}

static void expectEq(const vector<vector<int>>& a, const vector<vector<int>>& b) {
    assert(a == b);
}

int main() {
    Solution sol;

    {
        vector<vector<int>> m{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}};
        sol.setZeroes(m);
        expectEq(m, vector<vector<int>>({{1, 0, 1}, {0, 0, 0}, {1, 0, 1}}));
    }
    {
        vector<vector<int>> m{{0, 1, 2, 0}, {3, 4, 5, 2}, {1, 3, 1, 5}};
        sol.setZeroes(m);
        expectEq(m, vector<vector<int>>({{0, 0, 0, 0}, {0, 4, 5, 0}, {0, 3, 1, 0}}));
    }
    {
        vector<vector<int>> m{{1}};
        sol.setZeroes(m);
        expectEq(m, vector<vector<int>>({{1}}));
    }
    {
        vector<vector<int>> m{{0}};
        sol.setZeroes(m);
        expectEq(m, vector<vector<int>>({{0}}));
    }
    {
        vector<vector<int>> m{{1, 0, 3}};
        sol.setZeroes(m);
        expectEq(m, vector<vector<int>>({{0, 0, 0}}));
    }
    {
        vector<vector<int>> m{{1}, {0}, {3}};
        sol.setZeroes(m);
        expectEq(m, vector<vector<int>>({{0}, {0}, {0}}));
    }

    mt19937 rng(20260330);
    uniform_int_distribution<int> mDist(1, 30);
    uniform_int_distribution<int> nDist(1, 30);
    uniform_int_distribution<int> valDist(-3, 3);

    for (int tc = 0; tc < 2000; ++tc) {
        int m = mDist(rng);
        int n = nDist(rng);
        vector<vector<int>> a(m, vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) a[i][j] = valDist(rng);
        }

        auto want = bruteSetZeroes(a);
        sol.setZeroes(a);
        expectEq(a, want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

