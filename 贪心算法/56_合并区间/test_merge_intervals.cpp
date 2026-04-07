#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());

        vector<vector<int>> res;
        res.reserve(intervals.size());

        for (const auto& seg : intervals) {
            if (res.empty() || res.back()[1] < seg[0]) {
                res.push_back(seg);
            } else {
                res.back()[1] = max(res.back()[1], seg[1]);
            }
        }
        return res;
    }
};

static vector<vector<int>> bruteMerge(vector<vector<int>> intervals) {
    bool changed = true;
    while (changed) {
        changed = false;
        sort(intervals.begin(), intervals.end());
        vector<vector<int>> next;
        for (const auto& seg : intervals) {
            if (next.empty() || next.back()[1] < seg[0]) {
                next.push_back(seg);
            } else {
                next.back()[1] = max(next.back()[1], seg[1]);
                changed = true;
            }
        }
        intervals.swap(next);
    }
    return intervals;
}

static void normalize(vector<vector<int>>& a) {
    sort(a.begin(), a.end());
}

static void expectEq(vector<vector<int>> a, vector<vector<int>> b) {
    normalize(a);
    normalize(b);
    assert(a == b);
}

int main() {
    Solution sol;

    {
        vector<vector<int>> intervals{{1, 3}, {2, 6}, {8, 10}, {15, 18}};
        auto got = sol.merge(intervals);
        expectEq(got, vector<vector<int>>({{1, 6}, {8, 10}, {15, 18}}));
    }
    {
        vector<vector<int>> intervals{{1, 4}, {4, 5}};
        auto got = sol.merge(intervals);
        expectEq(got, vector<vector<int>>({{1, 5}}));
    }
    {
        vector<vector<int>> intervals{{4, 7}, {1, 4}};
        auto got = sol.merge(intervals);
        expectEq(got, vector<vector<int>>({{1, 7}}));
    }
    {
        vector<vector<int>> intervals{{1, 1}, {2, 2}, {3, 3}};
        auto got = sol.merge(intervals);
        expectEq(got, vector<vector<int>>({{1, 1}, {2, 2}, {3, 3}}));
    }

    mt19937 rng(20260330);
    uniform_int_distribution<int> nDist(1, 200);
    uniform_int_distribution<int> coordDist(0, 20);

    for (int tc = 0; tc < 3000; ++tc) {
        int n = nDist(rng);
        vector<vector<int>> intervals;
        intervals.reserve(n);
        for (int i = 0; i < n; ++i) {
            int a = coordDist(rng);
            int b = coordDist(rng);
            if (a > b) swap(a, b);
            intervals.push_back({a, b});
        }

        auto intervalsCopy = intervals;
        auto got = sol.merge(intervalsCopy);
        auto want = bruteMerge(intervals);
        expectEq(got, want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

