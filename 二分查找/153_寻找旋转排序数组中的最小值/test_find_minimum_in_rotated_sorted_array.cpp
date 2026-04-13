#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    int findMin(vector<int>& nums) {
        int n = (int)nums.size();
        int l = 0, r = n - 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] > nums[r]) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        return nums[l];
    }
};

static int bruteMin(const vector<int>& nums) {
    return *min_element(nums.begin(), nums.end());
}

static vector<int> rotateK(const vector<int>& a, int k) {
    int n = (int)a.size();
    vector<int> b(n);
    for (int i = 0; i < n; ++i) b[(i + k) % n] = a[i];
    return b;
}

int main() {
    Solution sol;

    {
        vector<int> nums{3, 4, 5, 1, 2};
        assert(sol.findMin(nums) == 1);
    }
    {
        vector<int> nums{4, 5, 6, 7, 0, 1, 2};
        assert(sol.findMin(nums) == 0);
    }
    {
        vector<int> nums{11, 13, 15, 17};
        assert(sol.findMin(nums) == 11);
    }
    {
        vector<int> nums{2, 1};
        assert(sol.findMin(nums) == 1);
    }
    {
        vector<int> nums{1};
        assert(sol.findMin(nums) == 1);
    }

    mt19937 rng(20260410);
    uniform_int_distribution<int> nDist(1, 200);
    uniform_int_distribution<int> valDist(-5000, 5000);

    for (int tc = 0; tc < 2000; ++tc) {
        int n = nDist(rng);
        vector<int> a;
        a.reserve(n);
        while ((int)a.size() < n) {
            int v = valDist(rng);
            bool ok = true;
            for (int x : a) {
                if (x == v) {
                    ok = false;
                    break;
                }
            }
            if (ok) a.push_back(v);
        }
        sort(a.begin(), a.end());
        uniform_int_distribution<int> kDist(0, n - 1);
        int k = kDist(rng);
        vector<int> nums = rotateK(a, k);
        vector<int> copy = nums;
        int got = sol.findMin(copy);
        int want = bruteMin(nums);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

