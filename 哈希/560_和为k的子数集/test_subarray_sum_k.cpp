#include <cassert>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

using namespace std;

class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<long long, int> cnt;
        cnt.reserve(nums.size() * 2 + 1);
        cnt[0] = 1;

        long long pref = 0;
        long long ans = 0;

        for (int x : nums) {
            pref += x;
            auto it = cnt.find(pref - k);
            if (it != cnt.end()) ans += it->second;
            cnt[pref]++;
        }

        return (int)ans;
    }
};

static int bruteSubarraySum(const vector<int>& nums, int k) {
    int n = (int)nums.size();
    int ans = 0;
    for (int i = 0; i < n; ++i) {
        int sum = 0;
        for (int j = i; j < n; ++j) {
            sum += nums[j];
            if (sum == k) ans++;
        }
    }
    return ans;
}

int main() {
    Solution sol;

    {
        vector<int> nums{1, 1, 1};
        assert(sol.subarraySum(nums, 2) == 2);
    }
    {
        vector<int> nums{1, 2, 3};
        assert(sol.subarraySum(nums, 3) == 2);
    }
    {
        vector<int> nums{1};
        assert(sol.subarraySum(nums, 0) == 0);
        assert(sol.subarraySum(nums, 1) == 1);
    }
    {
        vector<int> nums{0, 0, 0};
        assert(sol.subarraySum(nums, 0) == 6);
    }
    {
        vector<int> nums{-1, -1, 1};
        assert(sol.subarraySum(nums, 0) == 1);
    }

    mt19937 rng(20260328);
    uniform_int_distribution<int> nDist(1, 50);
    uniform_int_distribution<int> vDist(-5, 5);
    uniform_int_distribution<int> kDist(-10, 10);

    for (int t = 0; t < 3000; ++t) {
        int n = nDist(rng);
        vector<int> nums(n);
        for (int i = 0; i < n; ++i) nums[i] = vDist(rng);
        int k = kDist(rng);

        vector<int> numsCopy = nums;
        int got = sol.subarraySum(numsCopy, k);
        int want = bruteSubarraySum(nums, k);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

