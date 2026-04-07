#include <cassert>
#include <deque>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> dq;
        vector<int> res;
        res.reserve(nums.size() >= (size_t)k ? nums.size() - k + 1 : 0);

        for (int i = 0; i < (int)nums.size(); ++i) {
            while (!dq.empty() && dq.front() <= i - k) dq.pop_front();
            // 维护整体单调性，保持dq.front() 为当前窗口的最大值
            while (!dq.empty() && nums[dq.back()] <= nums[i]) dq.pop_back();
            // dp保存的内容为：当前窗口的最大值的索引
            dq.push_back(i);
            if (i >= k - 1) res.push_back(nums[dq.front()]);
        }
        return res;
    }
};

static vector<int> bruteMaxSlidingWindow(const vector<int>& nums, int k) {
    vector<int> res;
    int n = (int)nums.size();
    res.reserve(n - k + 1);
    for (int i = 0; i + k <= n; ++i) {
        int mx = nums[i];
        for (int j = i; j < i + k; ++j) mx = max(mx, nums[j]);
        res.push_back(mx);
    }
    return res;
}

static void expectEq(const vector<int>& a, const vector<int>& b) {
    assert(a == b);
}

int main() {
    Solution sol;

    {
        vector<int> nums{1, 3, -1, -3, 5, 3, 6, 7};
        expectEq(sol.maxSlidingWindow(nums, 3), vector<int>({3, 3, 5, 5, 6, 7}));
    }
    {
        vector<int> nums{1};
        expectEq(sol.maxSlidingWindow(nums, 1), vector<int>({1}));
    }
    {
        vector<int> nums{9, 8, 7, 6};
        expectEq(sol.maxSlidingWindow(nums, 4), vector<int>({9}));
    }
    {
        vector<int> nums{4, 4, 4, 4};
        expectEq(sol.maxSlidingWindow(nums, 2), vector<int>({4, 4, 4}));
    }
    {
        vector<int> nums{-7, -8, 7, 5, 7, 1, 6, 0};
        expectEq(sol.maxSlidingWindow(nums, 3), bruteMaxSlidingWindow(nums, 3));
    }

    mt19937 rng(20260330);
    uniform_int_distribution<int> nDist(1, 200);
    uniform_int_distribution<int> valDist(-20, 20);

    for (int t = 0; t < 3000; ++t) {
        int n = nDist(rng);
        vector<int> nums(n);
        for (int i = 0; i < n; ++i) nums[i] = valDist(rng);
        uniform_int_distribution<int> kDist(1, n);
        int k = kDist(rng);

        vector<int> numsCopy = nums;
        auto got = sol.maxSlidingWindow(numsCopy, k);
        auto want = bruteMaxSlidingWindow(nums, k);
        expectEq(got, want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

