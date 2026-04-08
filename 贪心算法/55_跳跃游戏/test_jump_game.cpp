#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    bool canJump(vector<int>& nums) {
        int far = 0;
        int n = (int)nums.size();
        for (int i = 0; i < n; ++i) {
            if (i > far) return false;
            far = max(far, i + nums[i]);
            if (far >= n - 1) return true;
        }
        return true;
    }
};

static bool bruteCanJump(const vector<int>& nums) {
    int n = (int)nums.size();
    vector<char> can(n, 0);
    can[0] = 1;
    for (int i = 0; i < n; ++i) {
        if (!can[i]) continue;
        for (int step = 1; step <= nums[i] && i + step < n; ++step) {
            can[i + step] = 1;
        }
    }
    return can[n - 1] != 0;
}

int main() {
    Solution sol;

    {
        vector<int> nums{2, 3, 1, 1, 4};
        assert(sol.canJump(nums) == true);
    }
    {
        vector<int> nums{3, 2, 1, 0, 4};
        assert(sol.canJump(nums) == false);
    }
    {
        vector<int> nums{0};
        assert(sol.canJump(nums) == true);
    }
    {
        vector<int> nums{1, 0};
        assert(sol.canJump(nums) == true);
    }
    {
        vector<int> nums{0, 1};
        assert(sol.canJump(nums) == false);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(1, 200);
    uniform_int_distribution<int> valDist(0, 30);

    for (int tc = 0; tc < 5000; ++tc) {
        int n = nDist(rng);
        vector<int> nums(n);
        for (int i = 0; i < n; ++i) nums[i] = valDist(rng);
        vector<int> copy = nums;
        bool got = sol.canJump(copy);
        bool want = bruteCanJump(nums);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

