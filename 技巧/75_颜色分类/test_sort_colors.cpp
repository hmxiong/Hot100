#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    void sortColors(vector<int>& nums) {
        int left = 0;
        int i = 0;
        int right = (int)nums.size() - 1;

        while (i <= right) {
            if (nums[i] == 0) {
                swap(nums[i], nums[left]);
                ++left;
                ++i;
            } else if (nums[i] == 2) {
                swap(nums[i], nums[right]);
                --right;
            } else {
                ++i;
            }
        }
    }
};

static vector<int> bruteSorted(vector<int> nums) {
    sort(nums.begin(), nums.end());
    return nums;
}

static void expectEq(const vector<int>& a, const vector<int>& b) {
    assert(a == b);
}

int main() {
    Solution sol;

    {
        vector<int> nums{2, 0, 2, 1, 1, 0};
        sol.sortColors(nums);
        expectEq(nums, vector<int>({0, 0, 1, 1, 2, 2}));
    }
    {
        vector<int> nums{2, 0, 1};
        sol.sortColors(nums);
        expectEq(nums, vector<int>({0, 1, 2}));
    }
    {
        vector<int> nums{0};
        sol.sortColors(nums);
        expectEq(nums, vector<int>({0}));
    }
    {
        vector<int> nums{1};
        sol.sortColors(nums);
        expectEq(nums, vector<int>({1}));
    }
    {
        vector<int> nums{2};
        sol.sortColors(nums);
        expectEq(nums, vector<int>({2}));
    }
    {
        vector<int> nums{0, 0, 0, 0};
        sol.sortColors(nums);
        expectEq(nums, vector<int>({0, 0, 0, 0}));
    }
    {
        vector<int> nums{2, 2, 2, 2};
        sol.sortColors(nums);
        expectEq(nums, vector<int>({2, 2, 2, 2}));
    }
    {
        vector<int> nums{1, 1, 1, 1};
        sol.sortColors(nums);
        expectEq(nums, vector<int>({1, 1, 1, 1}));
    }

    mt19937 rng(20260408);
    uniform_int_distribution<int> nDist(1, 300);
    uniform_int_distribution<int> valDist(0, 2);

    for (int tc = 0; tc < 5000; ++tc) {
        int n = nDist(rng);
        vector<int> nums(n);
        for (int i = 0; i < n; ++i) nums[i] = valDist(rng);

        vector<int> want = bruteSorted(nums);
        sol.sortColors(nums);
        expectEq(nums, want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

