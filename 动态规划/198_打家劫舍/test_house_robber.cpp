#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int rob(vector<int>& nums) {
        int prev2 = 0;
        int prev1 = 0;
        for (int x : nums) {
            int cur = max(prev1, prev2 + x);
            prev2 = prev1;
            prev1 = cur;
        }
        return prev1;
    }
};

static int bruteRob(const vector<int>& nums) {
    int n = (int)nums.size();
    int best = 0;
    for (int mask = 0; mask < (1 << n); ++mask) {
        bool ok = true;
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            if ((mask >> i) & 1) {
                if (i > 0 && ((mask >> (i - 1)) & 1)) {
                    ok = false;
                    break;
                }
                sum += nums[i];
            }
        }
        if (ok) best = max(best, sum);
    }
    return best;
}

int main() {
    Solution sol;

    {
        vector<int> nums{1, 2, 3, 1};
        assert(sol.rob(nums) == 4);
    }
    {
        vector<int> nums{2, 7, 9, 3, 1};
        assert(sol.rob(nums) == 12);
    }
    {
        vector<int> nums{0};
        assert(sol.rob(nums) == 0);
    }
    {
        vector<int> nums{400, 0, 400};
        assert(sol.rob(nums) == 800);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(1, 20);
    uniform_int_distribution<int> valDist(0, 20);

    for (int tc = 0; tc < 2000; ++tc) {
        int n = nDist(rng);
        vector<int> nums(n);
        for (int i = 0; i < n; ++i) nums[i] = valDist(rng);
        vector<int> copy = nums;
        int got = sol.rob(copy);
        int want = bruteRob(nums);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

