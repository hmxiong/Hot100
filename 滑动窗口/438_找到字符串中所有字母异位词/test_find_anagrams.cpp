#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int n = (int)s.size(), m = (int)p.size();
        vector<int> res;
        if (n < m) return res;

        int diff[26] = {0};
        for (char c : p) diff[c - 'a']++;

        int mismatch = 0;
        for (int i = 0; i < 26; ++i) {
            if (diff[i] != 0) mismatch++;
        }
        auto add = [&](int idx, int delta) {
            int before = diff[idx];
            int after = before + delta;
            if (before == 0 && after != 0) mismatch++;
            if (before != 0 && after == 0) mismatch--;
            diff[idx] = after;
        };

        for (int i = 0; i < m; ++i) add(s[i] - 'a', -1);
        if (mismatch == 0) res.push_back(0);

        for (int r = m; r < n; ++r) {
            add(s[r] - 'a', -1);
            add(s[r - m] - 'a', +1);
            if (mismatch == 0) res.push_back(r - m + 1);
        }

        return res;
    }
};

static vector<int> bruteFindAnagrams(const string& s, const string& p) {
    vector<int> res;
    int n = (int)s.size(), m = (int)p.size();
    if (n < m) return res;

    array<int, 26> need{};
    for (char c : p) need[c - 'a']++;

    for (int i = 0; i + m <= n; ++i) {
        array<int, 26> cnt{};
        for (int j = 0; j < m; ++j) cnt[s[i + j] - 'a']++;
        if (cnt == need) res.push_back(i);
    }
    return res;
}

static void expectEq(vector<int> a, vector<int> b) {
    sort(a.begin(), a.end());
    sort(b.begin(), b.end());
    assert(a == b);
}

int main() {
    Solution sol;

    expectEq(sol.findAnagrams("cbaebabacd", "abc"), vector<int>({0, 6}));
    expectEq(sol.findAnagrams("abab", "ab"), vector<int>({0, 1, 2}));
    expectEq(sol.findAnagrams("", "a"), vector<int>({}));
    expectEq(sol.findAnagrams("a", "a"), vector<int>({0}));
    expectEq(sol.findAnagrams("aaaaa", "aa"), vector<int>({0, 1, 2, 3}));
    expectEq(sol.findAnagrams("baa", "aa"), vector<int>({1}));
    expectEq(sol.findAnagrams("abc", "abcd"), vector<int>({}));

    mt19937 rng(123456);
    uniform_int_distribution<int> lenS(1, 30);
    uniform_int_distribution<int> lenP(1, 10);
    uniform_int_distribution<int> ch(0, 2);

    for (int t = 0; t < 2000; ++t) {
        int n = lenS(rng);
        int m = min(lenP(rng), n);

        string s;
        string p;
        s.reserve(n);
        p.reserve(m);

        for (int i = 0; i < n; ++i) s.push_back(char('a' + ch(rng)));
        for (int i = 0; i < m; ++i) p.push_back(char('a' + ch(rng)));

        auto got = sol.findAnagrams(s, p);
        auto want = bruteFindAnagrams(s, p);
        expectEq(got, want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}
