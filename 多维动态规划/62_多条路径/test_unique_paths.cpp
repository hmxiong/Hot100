#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<long long> dp(n, 1);
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) dp[j] += dp[j - 1];
        }
        return (int)dp[n - 1];
    }
};

static long long combUniquePathsSmall(int m, int n) {
    int a = m - 1;
    int b = n - 1;
    int k = min(a, b);
    int total = a + b;

    vector<long long> numerators;
    numerators.reserve(k);
    for (int x = total - k + 1; x <= total; ++x) numerators.push_back(x);

    for (int d = 2; d <= k; ++d) {
        long long denom = d;
        for (int i = 0; i < (int)numerators.size() && denom > 1; ++i) {
            long long g = gcd(numerators[i], denom);
            if (g > 1) {
                numerators[i] /= g;
                denom /= g;
            }
        }
        assert(denom == 1);
    }

    long long res = 1;
    for (long long x : numerators) res *= x;
    return res;
}

int main() {
    Solution sol;

    assert(sol.uniquePaths(3, 7) == 28);
    assert(sol.uniquePaths(3, 2) == 3);
    assert(sol.uniquePaths(7, 3) == 28);
    assert(sol.uniquePaths(3, 3) == 6);
    assert(sol.uniquePaths(1, 1) == 1);
    assert(sol.uniquePaths(1, 100) == 1);
    assert(sol.uniquePaths(100, 1) == 1);

    mt19937 rng(20260408);
    uniform_int_distribution<int> dist(1, 15);
    for (int tc = 0; tc < 2000; ++tc) {
        int m = dist(rng);
        int n = dist(rng);
        int got = sol.uniquePaths(m, n);
        long long want = combUniquePathsSmall(m, n);
        assert((long long)got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

