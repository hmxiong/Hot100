#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int numSquares(int n) {
        vector<int> squares;
        for (int i = 1; i * i <= n; ++i) squares.push_back(i * i);

        const int INF = 1e9;
        vector<int> dp(n + 1, INF);
        dp[0] = 0;
        for (int x = 1; x <= n; ++x) {
            for (int s : squares) {
                if (s > x) break;
                dp[x] = min(dp[x], dp[x - s] + 1);
            }
        }
        return dp[n];
    }
};

static int bruteNumSquares(int n) {
    vector<int> squares;
    for (int i = 1; i * i <= n; ++i) squares.push_back(i * i);
    vector<int> dist(n + 1, -1);
    dist[0] = 0;
    vector<int> q;
    q.push_back(0);
    for (int qi = 0; qi < (int)q.size(); ++qi) {
        int cur = q[qi];
        for (int s : squares) {
            int nxt = cur + s;
            if (nxt > n) break;
            if (dist[nxt] != -1) continue;
            dist[nxt] = dist[cur] + 1;
            q.push_back(nxt);
        }
    }
    return dist[n];
}

int main() {
    Solution sol;

    assert(sol.numSquares(12) == 3);
    assert(sol.numSquares(13) == 2);
    assert(sol.numSquares(1) == 1);
    assert(sol.numSquares(2) == 2);
    assert(sol.numSquares(3) == 3);
    assert(sol.numSquares(4) == 1);
    assert(sol.numSquares(9999) >= 1 && sol.numSquares(9999) <= 4);

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(1, 10000);
    for (int tc = 0; tc < 2000; ++tc) {
        int n = nDist(rng);
        int got = sol.numSquares(n);
        int want = bruteNumSquares(n);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

