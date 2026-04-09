#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int slow = nums[0];
        int fast = nums[nums[0]];
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        slow = 0;
        while (slow != fast) {
            slow = nums[slow];
            fast = nums[fast];
        }
        return slow;
    }
};

static int bruteFindDuplicate(vector<int> nums) {
    sort(nums.begin(), nums.end());
    for (int i = 1; i < (int)nums.size(); ++i) if (nums[i] == nums[i - 1]) return nums[i];
    return -1;
}

static vector<int> genOneDuplicate(mt19937& rng, int n, int dup) {
    vector<int> a;
    a.reserve(n + 1);
    for (int v = 1; v <= n; ++v) a.push_back(v);
    a.push_back(dup);
    shuffle(a.begin(), a.end(), rng);
    return a;
}

int main() {
    Solution sol;

    {
        vector<int> nums{1, 3, 4, 2, 2};
        assert(sol.findDuplicate(nums) == 2);
    }
    {
        vector<int> nums{3, 1, 3, 4, 2};
        assert(sol.findDuplicate(nums) == 3);
    }
    {
        vector<int> nums{2, 2};
        assert(sol.findDuplicate(nums) == 2);
    }
    {
        vector<int> nums{2, 5, 9, 6, 9, 3, 8, 9, 7, 1};
        assert(sol.findDuplicate(nums) == 9);
    }

    mt19937 rng(20260409);
    uniform_int_distribution<int> nDist(1, 5000);
    for (int tc = 0; tc < 1000; ++tc) {
        int n = nDist(rng);
        uniform_int_distribution<int> dDist(1, n);
        int dup = dDist(rng);
        vector<int> nums = genOneDuplicate(rng, n, dup);
        int got = sol.findDuplicate(nums);
        int want = bruteFindDuplicate(nums);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

