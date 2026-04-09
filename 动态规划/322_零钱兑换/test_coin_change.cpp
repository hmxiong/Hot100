#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        const int INF = 1e9;
        vector<int> dp(amount + 1, INF);
        dp[0] = 0;
        for (int c : coins) {
            for (int x = c; x <= amount; ++x) {
                dp[x] = min(dp[x], dp[x - c] + 1);
            }
        }
        return dp[amount] >= INF ? -1 : dp[amount];
    }
};

static int bruteCoinChangeBFS(const vector<int>& coins, int amount) {
    if (amount == 0) return 0;
    vector<int> dist(amount + 1, -1);
    queue<int> q;
    dist[0] = 0;
    q.push(0);
    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        for (int c : coins) {
            int nxt = cur + c;
            if (nxt > amount) continue;
            if (dist[nxt] != -1) continue;
            dist[nxt] = dist[cur] + 1;
            if (nxt == amount) return dist[nxt];
            q.push(nxt);
        }
    }
    return -1;
}

int main() {
    Solution sol;

    {
        vector<int> coins{1, 2, 5};
        assert(sol.coinChange(coins, 11) == 3);
    }
    {
        vector<int> coins{2};
        assert(sol.coinChange(coins, 3) == -1);
    }
    {
        vector<int> coins{1};
        assert(sol.coinChange(coins, 0) == 0);
        assert(sol.coinChange(coins, 2) == 2);
    }
    {
        vector<int> coins{2, 5, 10, 1};
        assert(sol.coinChange(coins, 27) == 4);
    }
    {
        vector<int> coins{186, 419, 83, 408};
        assert(sol.coinChange(coins, 6249) == 20);
    }

    mt19937 rng(20260409);
    uniform_int_distribution<int> nDist(1, 6);
    uniform_int_distribution<int> valDist(1, 50);
    uniform_int_distribution<int> amtDist(0, 200);

    for (int tc = 0; tc < 3000; ++tc) {
        int n = nDist(rng);
        vector<int> coins;
        coins.reserve(n);
        for (int i = 0; i < n; ++i) coins.push_back(valDist(rng));
        int amount = amtDist(rng);
        vector<int> coinsCopy = coins;
        int got = sol.coinChange(coinsCopy, amount);
        int want = bruteCoinChangeBFS(coins, amount);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

