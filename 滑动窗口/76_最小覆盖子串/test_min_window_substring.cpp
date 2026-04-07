#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <string>

using namespace std;

class Solution {
public:
    string minWindow(string s, string t) {
        if (t.empty() || s.empty()) return "";

        int need[256] = {0};
        for (unsigned char c : t) need[c]++;

        int missing = (int)t.size();
        int bestLen = (int)s.size() + 1;
        int bestL = 0;

        int l = 0;
        for (int r = 0; r < (int)s.size(); ++r) {
            unsigned char cr = (unsigned char)s[r];
            if (need[cr] > 0) missing--;
            need[cr]--;

            while (missing == 0) {
                int len = r - l + 1;
                if (len < bestLen) {
                    bestLen = len;
                    bestL = l;
                }

                unsigned char cl = (unsigned char)s[l];
                need[cl]++;
                if (need[cl] > 0) missing++;
                l++;
            }
        }

        if (bestLen > (int)s.size()) return "";
        return s.substr(bestL, bestLen);
    }
};

static bool covers(const string& sub, const string& t) {
    int cnt[256] = {0};
    for (unsigned char c : sub) cnt[c]++;
    for (unsigned char c : t) {
        if (cnt[c] == 0) return false;
        cnt[c]--;
    }
    return true;
}

static string bruteMinWindow(const string& s, const string& t) {
    int n = (int)s.size();
    int bestLen = n + 1;
    int bestL = 0;
    for (int l = 0; l < n; ++l) {
        for (int r = l; r < n; ++r) {
            int len = r - l + 1;
            if (len >= bestLen) continue;
            if (covers(s.substr(l, len), t)) {
                bestLen = len;
                bestL = l;
            }
        }
    }
    if (bestLen == n + 1) return "";
    return s.substr(bestL, bestLen);
}

int main() {
    Solution sol;

    assert(sol.minWindow("ADOBECODEBANC", "ABC") == "BANC");
    assert(sol.minWindow("a", "a") == "a");
    assert(sol.minWindow("a", "aa") == "");
    assert(sol.minWindow("aa", "aa") == "aa");
    assert(sol.minWindow("ab", "b") == "b");
    assert(sol.minWindow("bba", "ab") == "ba");

    mt19937 rng(20260330);
    uniform_int_distribution<int> lenS(1, 18);
    uniform_int_distribution<int> lenT(1, 6);
    uniform_int_distribution<int> ch(0, 3);
    const string alphabet = "ABCD";

    for (int tc = 0; tc < 3000; ++tc) {
        int n = lenS(rng);
        string s;
        s.reserve(n);
        for (int i = 0; i < n; ++i) s.push_back(alphabet[ch(rng)]);

        int m = lenT(rng);
        string t;
        t.reserve(m);
        for (int i = 0; i < m; ++i) t.push_back(alphabet[ch(rng)]);

        string got = sol.minWindow(s, t);
        string want = bruteMinWindow(s, t);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

