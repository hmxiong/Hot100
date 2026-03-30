#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        int n = (int)nums.size();
        if (n == 0) return;
        k %= n;
        if (k == 0) return;
        reverse(nums.begin(), nums.end());
        reverse(nums.begin(), nums.begin() + k);
        reverse(nums.begin() + k, nums.end());
    }
};

static vector<int> bruteRotate(vector<int> a, int k) {
    int n = (int)a.size();
    if (n == 0) return a;
    k %= n;
    vector<int> b(n);
    for (int i = 0; i < n; ++i) {
        int j = (i + k) % n;
        b[j] = a[i];
    }
    return b;
}

static void expectEq(const vector<int>& a, const vector<int>& b) {
    assert(a == b);
}

int main() {
    Solution sol;

    {
        vector<int> v{1,2,3,4,5,6,7};
        sol.rotate(v, 3);
        expectEq(v, vector<int>({5,6,7,1,2,3,4}));
    }
    {
        vector<int> v{-1,-100,3,99};
        sol.rotate(v, 2);
        expectEq(v, vector<int>({3,99,-1,-100}));
    }
    {
        vector<int> v{1};
        sol.rotate(v, 0);
        expectEq(v, vector<int>({1}));
    }
    {
        vector<int> v{1,2};
        sol.rotate(v, 2);
        expectEq(v, vector<int>({1,2}));
    }

    mt19937 rng(20260330);
    uniform_int_distribution<int> nDist(0, 200);
    uniform_int_distribution<int> valDist(-1000, 1000);
    uniform_int_distribution<int> kDist(0, 1000);

    for (int t = 0; t < 3000; ++t) {
        int n = nDist(rng);
        vector<int> a(n);
        for (int i = 0; i < n; ++i) a[i] = valDist(rng);
        int k = kDist(rng);
        vector<int> b = a;
        sol.rotate(b, k);
        auto want = bruteRotate(a, k);
        expectEq(b, want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

