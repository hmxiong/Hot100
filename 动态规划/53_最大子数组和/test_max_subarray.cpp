#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int best = nums[0];
        int cur = nums[0];
        for (int i = 1; i < (int)nums.size(); ++i) {
            cur = max(nums[i], cur + nums[i]);
            best = max(best, cur);
        }
        return best;
    }
};

static int bruteMaxSubarray(const vector<int>& a) {
    int n = (int)a.size();
    int best = a[0];
    for (int i = 0; i < n; ++i) {
        int sum = 0;
        for (int j = i; j < n; ++j) {
            sum += a[j];
            best = max(best, sum);
        }
    }
    return best;
}

int main() {
    Solution sol;

    {
        vector<int> v{-2,1,-3,4,-1,2,1,-5,4};
        assert(sol.maxSubArray(v) == 6);
    }
    {
        vector<int> v{1};
        assert(sol.maxSubArray(v) == 1);
    }
    {
        vector<int> v{5,4,-1,7,8};
        assert(sol.maxSubArray(v) == 23);
    }
    {
        vector<int> v{-1,-2,-3};
        assert(sol.maxSubArray(v) == -1);
    }

    mt19937 rng(20260330);
    uniform_int_distribution<int> nDist(1, 200);
    uniform_int_distribution<int> valDist(-50, 50);

    for (int t = 0; t < 2000; ++t) {
        int n = nDist(rng);
        vector<int> v(n);
        for (int i = 0; i < n; ++i) v[i] = valDist(rng);
        vector<int> v2 = v;
        int got = sol.maxSubArray(v2);
        int want = bruteMaxSubarray(v);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

