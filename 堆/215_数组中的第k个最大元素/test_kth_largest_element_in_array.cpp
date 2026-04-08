#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class MinHeap {
public:
    void push(int x) {
        a.push_back(x);
        siftUp((int)a.size() - 1);
    }

    void pop() {
        int n = (int)a.size();
        swap(a[0], a[n - 1]);
        a.pop_back();
        if (!a.empty()) siftDown(0);
    }

    int top() const {
        return a[0];
    }

    int size() const {
        return (int)a.size();
    }

private:
    vector<int> a;

    void siftUp(int i) {
        while (i > 0) {
            int p = (i - 1) / 2;
            if (a[p] <= a[i]) break;
            swap(a[p], a[i]);
            i = p;
        }
    }

    void siftDown(int i) {
        int n = (int)a.size();
        while (true) {
            int l = i * 2 + 1;
            int r = i * 2 + 2;
            int best = i;
            if (l < n && a[l] < a[best]) best = l;
            if (r < n && a[r] < a[best]) best = r;
            if (best == i) break;
            swap(a[i], a[best]);
            i = best;
        }
    }
};

class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        MinHeap pq;
        for (int x : nums) {
            pq.push(x);
            if (pq.size() > k) pq.pop();
        }
        return pq.top();
    }
};

static int bruteKthLargest(vector<int> nums, int k) {
    sort(nums.begin(), nums.end());
    return nums[(int)nums.size() - k];
}

int main() {
    Solution sol;

    {
        vector<int> nums{3, 2, 1, 5, 6, 4};
        assert(sol.findKthLargest(nums, 2) == 5);
    }
    {
        vector<int> nums{3, 2, 3, 1, 2, 4, 5, 5, 6};
        assert(sol.findKthLargest(nums, 4) == 4);
    }
    {
        vector<int> nums{1};
        assert(sol.findKthLargest(nums, 1) == 1);
    }
    {
        vector<int> nums{2, 2, 2, 2};
        assert(sol.findKthLargest(nums, 3) == 2);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(1, 2000);
    uniform_int_distribution<int> valDist(-10000, 10000);

    for (int tc = 0; tc < 3000; ++tc) {
        int n = nDist(rng);
        vector<int> nums(n);
        for (int i = 0; i < n; ++i) nums[i] = valDist(rng);
        uniform_int_distribution<int> kDist(1, n);
        int k = kDist(rng);

        vector<int> copy = nums;
        int got = sol.findKthLargest(copy, k);
        int want = bruteKthLargest(nums, k);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}
