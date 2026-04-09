#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n = (int)nums.size();
        if (n <= 1) return;
        int i = n - 2;
        while (i >= 0 && nums[i] >= nums[i + 1]) --i;
        if (i >= 0) {
            int j = n - 1;
            while (nums[j] <= nums[i]) --j;
            swap(nums[i], nums[j]);
        }
        reverse(nums.begin() + (i + 1), nums.end());
    }
};

static vector<int> stdNext(vector<int> a) {
    next_permutation(a.begin(), a.end());
    return a;
}

static void expectEq(const vector<int>& a, const vector<int>& b) {
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) assert(a[i] == b[i]);
}

static string vecToStr(const vector<int>& v) {
    string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += ",";
        s += to_string(v[i]);
    }
    return s;
}

int main() {
    Solution sol;

    {
        vector<int> a{1, 2, 3};
        vector<int> b = a;
        sol.nextPermutation(a);
        next_permutation(b.begin(), b.end());
        expectEq(a, b);
    }
    {
        vector<int> a{3, 2, 1};
        vector<int> b = a;
        sol.nextPermutation(a);
        next_permutation(b.begin(), b.end());
        expectEq(a, b);
    }
    {
        vector<int> a{1, 1, 5};
        vector<int> b = a;
        sol.nextPermutation(a);
        next_permutation(b.begin(), b.end());
        expectEq(a, b);
    }
    {
        vector<int> a{1};
        vector<int> b = a;
        sol.nextPermutation(a);
        next_permutation(b.begin(), b.end());
        expectEq(a, b);
    }
    {
        vector<int> a{1, 5, 1};
        vector<int> b = a;
        sol.nextPermutation(a);
        next_permutation(b.begin(), b.end());
        expectEq(a, b);
    }

    mt19937 rng(20260409);
    uniform_int_distribution<int> nDist(1, 8);
    uniform_int_distribution<int> valDist(-3, 3);
    for (int tc = 0; tc < 5000; ++tc) {
        int n = nDist(rng);
        vector<int> a(n);
        for (int i = 0; i < n; ++i) a[i] = valDist(rng);
        vector<int> b = a;
        sol.nextPermutation(a);
        next_permutation(b.begin(), b.end());
        if (a != b) {
            cerr << "Mismatch: got [" << vecToStr(a) << "] want [" << vecToStr(b) << "]\n";
            assert(false);
        }
    }

    cout << "All tests passed." << "\n";
    return 0;
}

