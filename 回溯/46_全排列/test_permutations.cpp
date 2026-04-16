#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    void dfs(const vector<int>& nums, vector<char>& used, vector<int>& path, vector<vector<int>>& ans) {
        int n = (int)nums.size();
        if ((int)path.size() == n) {
            ans.push_back(path);
            return;
        }
        for (int i = 0; i < n; ++i) {
            if (used[i]) continue;
            used[i] = 1;
            path.push_back(nums[i]);
            dfs(nums, used, path, ans);
            path.pop_back();
            used[i] = 0;
        }
    }

    vector<vector<int>> permute(vector<int>& nums) {
        int n = (int)nums.size();
        vector<vector<int>> ans;
        vector<int> path;
        path.reserve(n);
        vector<char> used(n, 0);
        dfs(nums, used, path, ans);
        return ans;
    }
};

static vector<vector<int>> referencePermute(vector<int> nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> res;
    do {
        res.push_back(nums);
    } while (next_permutation(nums.begin(), nums.end()));
    return res;
}

int main() {
    Solution sol;
    {
        vector<int> nums{1, 2, 3};
        auto got = sol.permute(nums);
        auto expect = referencePermute(nums);
        sort(got.begin(), got.end());
        sort(expect.begin(), expect.end());
        assert(got == expect);
    }
    {
        vector<int> nums{0, 1};
        auto got = sol.permute(nums);
        auto expect = referencePermute(nums);
        sort(got.begin(), got.end());
        sort(expect.begin(), expect.end());
        assert(got == expect);
    }
    {
        vector<int> nums{1};
        auto got = sol.permute(nums);
        auto expect = referencePermute(nums);
        sort(got.begin(), got.end());
        sort(expect.begin(), expect.end());
        assert(got == expect);
    }

    mt19937 rng(20260416);
    for (int tc = 0; tc < 300; ++tc) {
        int n = 1 + (int)(rng() % 6);
        vector<int> nums;
        nums.reserve(n);
        while ((int)nums.size() < n) {
            int v = (int)(rng() % 21) - 10;
            bool ok = true;
            for (int x : nums) {
                if (x == v) {
                    ok = false;
                    break;
                }
            }
            if (ok) nums.push_back(v);
        }
        auto got = sol.permute(nums);
        auto expect = referencePermute(nums);
        sort(got.begin(), got.end());
        sort(expect.begin(), expect.end());
        assert(got == expect);
    }

    cout << "All tests passed." << "\n";
    return 0;
}
