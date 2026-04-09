#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> partitionLabels(const string& s) {
        int n = (int)s.size();
        vector<int> last(26, -1);
        for (int i = 0; i < n; ++i) last[s[i] - 'a'] = i;
        vector<int> res;
        int start = 0;
        int end = 0;
        for (int i = 0; i < n; ++i) {
            end = max(end, last[s[i] - 'a']);
            if (i == end) {
                res.push_back(end - start + 1);
                start = i + 1;
            }
        }
        return res;
    }
};

static vector<int> brutePartitionLabels(const string& s) {
    int n = (int)s.size();
    if (n == 0) return {};
    vector<int> first(26, n);
    vector<int> last(26, -1);
    for (int i = 0; i < n; ++i) {
        int c = s[i] - 'a';
        first[c] = min(first[c], i);
        last[c] = max(last[c], i);
    }
    vector<int> cuts;
    for (int i = 0; i < n - 1; ++i) {
        bool ok = true;
        for (int c = 0; c < 26; ++c) {
            if (first[c] <= i && last[c] > i) { ok = false; break; }
        }
        if (ok) cuts.push_back(i);
    }
    vector<int> res;
    int prev = -1;
    for (int x : cuts) {
        res.push_back(x - prev);
        prev = x;
    }
    res.push_back(n - 1 - prev);
    return res;
}

static void expectEq(const vector<int>& a, const vector<int>& b) {
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) assert(a[i] == b[i]);
}

int main() {
    Solution sol;

    expectEq(sol.partitionLabels("ababcbacadefegdehijhklij"), vector<int>({9, 7, 8}));
    expectEq(sol.partitionLabels("eccbbbbdec"), vector<int>({10}));
    expectEq(sol.partitionLabels("a"), vector<int>({1}));
    expectEq(sol.partitionLabels("aaaa"), vector<int>({4}));
    expectEq(sol.partitionLabels("abc"), vector<int>({1, 1, 1}));
    expectEq(sol.partitionLabels("abac"), vector<int>({3, 1}));

    mt19937 rng(20260409);
    uniform_int_distribution<int> nDist(1, 200);
    uniform_int_distribution<int> cDist(0, 25);
    for (int tc = 0; tc < 3000; ++tc) {
        int n = nDist(rng);
        string s;
        s.resize((size_t)n);
        for (int i = 0; i < n; ++i) s[i] = (char)('a' + cDist(rng));
        auto got = sol.partitionLabels(s);
        auto want = brutePartitionLabels(s);
        expectEq(got, want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

