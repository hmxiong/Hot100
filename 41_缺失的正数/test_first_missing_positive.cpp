#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = (int)nums.size();
        for (int i = 0; i < n; ++i) {
            while (nums[i] >= 1 && nums[i] <= n) {
                int v = nums[i];
                int pos = v - 1;
                if (nums[pos] == v) break;
                swap(nums[i], nums[pos]);
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1) return i + 1;
        }
        return n + 1;
    }
};

static int bruteFirstMissingPositive(const vector<int>& nums) {
    int n = (int)nums.size();
    for (int x = 1; x <= n + 1; ++x) {
        bool ok = false;
        for (int v : nums) {
            if (v == x) {
                ok = true;
                break;
            }
        }
        if (!ok) return x;
    }
    return n + 1;
}

int main() {
    Solution sol;

    {
        vector<int> nums{1, 2, 0};
        assert(sol.firstMissingPositive(nums) == 3);
    }
    {
        vector<int> nums{3, 4, -1, 1};
        assert(sol.firstMissingPositive(nums) == 2);
    }
    {
        vector<int> nums{7, 8, 9, 11, 12};
        assert(sol.firstMissingPositive(nums) == 1);
    }
    {
        vector<int> nums{1};
        assert(sol.firstMissingPositive(nums) == 2);
    }
    {
        vector<int> nums{2};
        assert(sol.firstMissingPositive(nums) == 1);
    }
    {
        vector<int> nums{1, 1};
        assert(sol.firstMissingPositive(nums) == 2);
    }

    mt19937 rng(20260330);
    uniform_int_distribution<int> nDist(1, 200);
    uniform_int_distribution<int> valDist(-50, 50);

    for (int tc = 0; tc < 4000; ++tc) {
        int n = nDist(rng);
        vector<int> nums(n);
        for (int i = 0; i < n; ++i) nums[i] = valDist(rng);

        vector<int> copy = nums;
        int got = sol.firstMissingPositive(copy);
        int want = bruteFirstMissingPositive(nums);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

