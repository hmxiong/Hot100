#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int jump(vector<int>& nums) {
        int n = (int)nums.size();
        if (n <= 1) return 0;

        int steps = 0;
        int currentEnd = 0;
        int farthest = 0;

        for (int i = 0; i < n - 1; ++i) {
            farthest = max(farthest, i + nums[i]);
            if (i == currentEnd) {
                ++steps;
                currentEnd = farthest;
            }
        }
        return steps;
    }

    vector<int> jumpPath(vector<int>& nums) {
        int n = (int)nums.size();
        vector<int> path;
        if (n <= 1) return path;
        int pos = 0;
        while (pos < n - 1) {
            int reach = nums[pos];
            if (pos + reach >= n - 1) {
                path.push_back((n - 1) - pos);
                break;
            }
            int bestNext = pos + 1;
            int bestReach = bestNext + nums[bestNext];
            int end = min(n - 1, pos + reach);
            for (int j = pos + 1; j <= end; ++j) {
                int r = j + nums[j];
                if (r > bestReach) {
                    bestReach = r;
                    bestNext = j;
                }
            }
            path.push_back(bestNext - pos);
            pos = bestNext;
        }
        return path;
    }
};

static int bruteMinJumps(const vector<int>& nums) {
    int n = (int)nums.size();
    vector<int> dist(n, -1);
    dist[0] = 0;

    vector<int> q;
    q.push_back(0);
    for (int qi = 0; qi < (int)q.size(); ++qi) {
        int cur = q[qi];
        int maxJump = nums[cur];
        for (int step = 1; step <= maxJump && cur + step < n; ++step) {
            int nxt = cur + step;
            if (dist[nxt] != -1) continue;
            dist[nxt] = dist[cur] + 1;
            q.push_back(nxt);
        }
    }
    return dist[n - 1];
}

static bool pathValid(const vector<int>& nums, const vector<int>& path) {
    int n = (int)nums.size();
    int pos = 0;
    for (int s : path) {
        if (s <= 0) return false;
        if (pos >= n) return false;
        if (s > nums[pos]) return false;
        pos += s;
    }
    return n > 0 && pos == n - 1;
}

int main() {
    Solution sol;

    {
        vector<int> nums{2, 3, 1, 1, 4};
        assert(sol.jump(nums) == 2);
        vector<int> p = sol.jumpPath(nums);
        assert((int)p.size() == 2);
        assert(pathValid(nums, p));
    }
    {
        vector<int> nums{2, 3, 0, 1, 4};
        assert(sol.jump(nums) == 2);
        vector<int> p = sol.jumpPath(nums);
        assert((int)p.size() == 2);
        assert(pathValid(nums, p));
    }
    {
        vector<int> nums{0};
        assert(sol.jump(nums) == 0);
        vector<int> p = sol.jumpPath(nums);
        assert(p.empty());
    }
    {
        vector<int> nums{1, 0};
        assert(sol.jump(nums) == 1);
        vector<int> p = sol.jumpPath(nums);
        assert((int)p.size() == 1);
        assert(pathValid(nums, p));
    }
    {
        vector<int> nums{4, 1, 1, 3, 1, 1, 1};
        assert(sol.jump(nums) == 2);
        vector<int> p = sol.jumpPath(nums);
        assert((int)p.size() == 2);
        assert(pathValid(nums, p));
    }

    mt19937 rng(20260408);
    uniform_int_distribution<int> nDist(1, 50);
    uniform_int_distribution<int> valDist(0, 10);

    for (int tc = 0; tc < 10000; ++tc) {
        int n = nDist(rng);
        vector<int> nums(n);
        for (int i = 0; i < n; ++i) nums[i] = valDist(rng);

        int want = bruteMinJumps(nums);
        if (want == -1) continue;

        vector<int> copy = nums;
        int got = sol.jump(copy);
        assert(got == want);
        vector<int> p = sol.jumpPath(nums);
        assert(pathValid(nums, p));
        assert((int)p.size() == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}
