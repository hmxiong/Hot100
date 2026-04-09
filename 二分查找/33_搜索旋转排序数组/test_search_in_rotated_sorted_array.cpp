#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int search(vector<int>& nums, int target) {
        int lo = 0;
        int hi = (int)nums.size() - 1;
        while (lo <= hi) {
            int mid = lo + (hi - lo) / 2;
            if (nums[mid] == target) return mid;

            if (nums[lo] <= nums[mid]) {
                if (nums[lo] <= target && target < nums[mid]) hi = mid - 1;
                else lo = mid + 1;
            } else {
                if (nums[mid] < target && target <= nums[hi]) lo = mid + 1;
                else hi = mid - 1;
            }
        }
        return -1;
    }
};

static int bruteSearch(const vector<int>& nums, int target) {
    for (int i = 0; i < (int)nums.size(); ++i) {
        if (nums[i] == target) return i;
    }
    return -1;
}

static vector<int> genSortedUnique(mt19937& rng, int n) {
    uniform_int_distribution<int> startDist(-20000, 20000);
    uniform_int_distribution<int> stepDist(1, 5);
    int cur = startDist(rng);
    vector<int> a;
    a.reserve(n);
    for (int i = 0; i < n; ++i) {
        cur += stepDist(rng);
        a.push_back(cur);
    }
    return a;
}

static vector<int> rotateLeft(const vector<int>& a, int k) {
    int n = (int)a.size();
    vector<int> b;
    b.reserve(n);
    for (int i = 0; i < n; ++i) b.push_back(a[(i + k) % n]);
    return b;
}

int main() {
    Solution sol;

    {
        vector<int> nums{4, 5, 6, 7, 0, 1, 2};
        assert(sol.search(nums, 0) == 4);
        assert(sol.search(nums, 3) == -1);
    }
    {
        vector<int> nums{1};
        assert(sol.search(nums, 0) == -1);
        assert(sol.search(nums, 1) == 0);
    }
    {
        vector<int> nums{1, 3};
        assert(sol.search(nums, 1) == 0);
        assert(sol.search(nums, 3) == 1);
        assert(sol.search(nums, 2) == -1);
    }
    {
        vector<int> nums{3, 1};
        assert(sol.search(nums, 3) == 0);
        assert(sol.search(nums, 1) == 1);
        assert(sol.search(nums, 2) == -1);
    }

    mt19937 rng(20260409);
    uniform_int_distribution<int> nDist(1, 5000);
    uniform_int_distribution<int> kDist(0, 4999);
    uniform_int_distribution<int> pickDist(0, 4999);
    uniform_int_distribution<int> extraDist(-30000, 30000);

    for (int tc = 0; tc < 1000; ++tc) {
        int n = nDist(rng);
        vector<int> base = genSortedUnique(rng, n);
        int k = kDist(rng) % n;
        vector<int> nums = rotateLeft(base, k);

        for (int q = 0; q < 30; ++q) {
            int target;
            if (q < 15) {
                target = base[pickDist(rng) % n];
            } else {
                target = extraDist(rng);
            }
            vector<int> copy = nums;
            int got = sol.search(copy, target);
            int want = bruteSearch(nums, target);
            assert(got == want);
        }
    }

    cout << "All tests passed." << "\n";
    return 0;
}

