#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int left = lowerBound(nums, target);
        if (left == (int)nums.size() || nums[left] != target) return {-1, -1};
        int right = lowerBound(nums, target + 1) - 1;
        return {left, right};
    }

private:
    static int lowerBound(const vector<int>& nums, int target) {
        int lo = 0;
        int hi = (int)nums.size();
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
};

static vector<int> brute(const vector<int>& nums, int target) {
    int l = -1;
    int r = -1;
    for (int i = 0; i < (int)nums.size(); ++i) {
        if (nums[i] == target) {
            if (l == -1) l = i;
            r = i;
        }
    }
    return {l, r};
}

static void expectEq(const vector<int>& a, const vector<int>& b) {
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) assert(a[i] == b[i]);
}

static vector<int> genSortedNums(mt19937& rng, int n) {
    uniform_int_distribution<int> stepDist(0, 3);
    uniform_int_distribution<int> startDist(-50, 50);
    int cur = startDist(rng);
    vector<int> nums;
    nums.reserve(n);
    for (int i = 0; i < n; ++i) {
        cur += stepDist(rng);
        nums.push_back(cur);
    }
    return nums;
}

int main() {
    Solution sol;

    {
        vector<int> nums{5, 7, 7, 8, 8, 10};
        expectEq(sol.searchRange(nums, 8), vector<int>({3, 4}));
        expectEq(sol.searchRange(nums, 6), vector<int>({-1, -1}));
    }
    {
        vector<int> nums;
        expectEq(sol.searchRange(nums, 0), vector<int>({-1, -1}));
    }
    {
        vector<int> nums{1};
        expectEq(sol.searchRange(nums, 1), vector<int>({0, 0}));
        expectEq(sol.searchRange(nums, 2), vector<int>({-1, -1}));
    }
    {
        vector<int> nums{2, 2, 2, 2};
        expectEq(sol.searchRange(nums, 2), vector<int>({0, 3}));
        expectEq(sol.searchRange(nums, 3), vector<int>({-1, -1}));
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 2000);
    uniform_int_distribution<int> targetDist(-60, 60);

    for (int tc = 0; tc < 2000; ++tc) {
        int n = nDist(rng);
        vector<int> nums = genSortedNums(rng, n);
        vector<int> copy = nums;
        for (int q = 0; q < 50; ++q) {
            int target = targetDist(rng);
            vector<int> got = sol.searchRange(copy, target);
            vector<int> want = brute(nums, target);
            expectEq(got, want);
        }
    }

    cout << "All tests passed." << "\n";
    return 0;
}

