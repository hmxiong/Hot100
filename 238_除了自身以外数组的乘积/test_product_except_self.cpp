#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = (int)nums.size();
        vector<int> res(n, 1);
        // nums:[1,2,3,4]
        
        long long prefix = 1;
        for (int i = 0; i < n; ++i) {
            res[i] = (int)prefix;
            prefix *= nums[i];
        }
        // res:[1,1,2,6]

        long long suffix = 1;
        for (int i = n - 1; i >= 0; --i) {
            res[i] = (int)(res[i] * suffix);
            suffix *= nums[i];
        }
        // suffix:[24,12,4,1]
        
        // res:[24,12,8,6]

        return res;
    }
};

static vector<int> bruteProductExceptSelf(const vector<int>& nums) {
    int n = (int)nums.size();
    vector<int> res(n, 1);
    for (int i = 0; i < n; ++i) {
        long long prod = 1;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            prod *= nums[j];
        }
        res[i] = (int)prod;
    }
    return res;
}

static void expectEq(const vector<int>& a, const vector<int>& b) {
    assert(a == b);
}

int main() {
    Solution sol;

    {
        vector<int> nums{1, 2, 3, 4};
        auto got = sol.productExceptSelf(nums);
        expectEq(got, vector<int>({24, 12, 8, 6}));
    }
    {
        vector<int> nums{-1, 1, 0, -3, 3};
        auto got = sol.productExceptSelf(nums);
        expectEq(got, vector<int>({0, 0, 9, 0, 0}));
    }
    {
        vector<int> nums{0, 0, 2, 3};
        auto got = sol.productExceptSelf(nums);
        expectEq(got, vector<int>({0, 0, 0, 0}));
    }
    {
        vector<int> nums{5, 0, 2, 0};
        auto got = sol.productExceptSelf(nums);
        expectEq(got, vector<int>({0, 0, 0, 0}));
    }
    {
        vector<int> nums{-2, -3};
        auto got = sol.productExceptSelf(nums);
        expectEq(got, vector<int>({-3, -2}));
    }

    mt19937 rng(20260330);
    uniform_int_distribution<int> nDist(2, 40);
    uniform_int_distribution<int> valDist(-5, 5);

    for (int tc = 0; tc < 5000; ++tc) {
        int n = nDist(rng);
        vector<int> nums(n);
        for (int i = 0; i < n; ++i) nums[i] = valDist(rng);

        vector<int> numsCopy = nums;
        auto got = sol.productExceptSelf(numsCopy);
        auto want = bruteProductExceptSelf(nums);
        expectEq(got, want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

