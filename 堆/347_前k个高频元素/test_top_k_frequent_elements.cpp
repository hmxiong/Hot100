#include <algorithm>
#include <cassert>
#include <iostream>
#include <queue>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> cnt;
        for (int x : nums) cnt[x]++;
        using P = pair<int, int>;
        priority_queue<P, vector<P>, greater<P>> pq;
        for (auto& e : cnt) {
            int val = e.first;
            int c = e.second;
            pq.push({c, val});
            if ((int)pq.size() > k) pq.pop();
        }
        vector<int> res;
        res.reserve(k);
        while (!pq.empty()) {
            res.push_back(pq.top().second);
            pq.pop();
        }
        return res;
    }
};

static vector<pair<int,int>> normalizePairs(const vector<int>& vals, const unordered_map<int,int>& cnt) {
    vector<pair<int,int>> ps;
    ps.reserve(vals.size());
    for (int v : vals) ps.push_back({cnt.at(v), v});
    sort(ps.begin(), ps.end(), [](const pair<int,int>& a, const pair<int,int>& b){
        if (a.first != b.first) return a.first > b.first;
        return a.second > b.second;
    });
    return ps;
}

static vector<int> bruteTopK(const vector<int>& nums, int k) {
    unordered_map<int, int> cnt;
    for (int x : nums) cnt[x]++;
    vector<pair<int,int>> ps;
    ps.reserve(cnt.size());
    for (auto& e : cnt) ps.push_back({e.second, e.first});
    sort(ps.begin(), ps.end(), [](const pair<int,int>& a, const pair<int,int>& b){
        if (a.first != b.first) return a.first > b.first;
        return a.second > b.second;
    });
    vector<int> res;
    for (int i = 0; i < k && i < (int)ps.size(); ++i) res.push_back(ps[i].second);
    return res;
}

static void expectSameTopK(vector<int> got, const vector<int>& nums, int k) {
    unordered_map<int,int> cnt;
    for (int x : nums) cnt[x]++;
    auto want = bruteTopK(nums, k);
    auto A = normalizePairs(got, cnt);
    auto B = normalizePairs(want, cnt);
    assert(A.size() == B.size());
    for (size_t i = 0; i < A.size(); ++i) assert(A[i] == B[i]);
}

int main() {
    Solution sol;

    {
        vector<int> nums{1,1,1,2,2,3};
        int k = 2;
        auto got = sol.topKFrequent(nums, k);
        expectSameTopK(got, nums, k);
    }
    {
        vector<int> nums{1};
        int k = 1;
        auto got = sol.topKFrequent(nums, k);
        expectSameTopK(got, nums, k);
    }
    {
        vector<int> nums{4,1,-1,2,-1,2,3};
        int k = 2;
        auto got = sol.topKFrequent(nums, k);
        expectSameTopK(got, nums, k);
    }
    {
        vector<int> nums{5,5,5,5,6,6,7,8,8};
        int k = 3;
        auto got = sol.topKFrequent(nums, k);
        expectSameTopK(got, nums, k);
    }

    mt19937 rng(20260409);
    uniform_int_distribution<int> nDist(1, 300);
    uniform_int_distribution<int> valDist(-50, 50);
    for (int tc = 0; tc < 3000; ++tc) {
        int n = nDist(rng);
        vector<int> nums(n);
        for (int i = 0; i < n; ++i) nums[i] = valDist(rng);
        unordered_map<int,int> cnt;
        for (int x : nums) cnt[x]++;
        int uniq = (int)cnt.size();
        uniform_int_distribution<int> kDist(1, max(1, uniq));
        int k = kDist(rng);
        auto got = sol.topKFrequent(nums, k);
        expectSameTopK(got, nums, k);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

