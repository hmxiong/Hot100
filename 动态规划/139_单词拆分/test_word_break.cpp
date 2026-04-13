#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = (int)s.size();
        unordered_set<string> dict(wordDict.begin(), wordDict.end());
        int minL = 1, maxL = 0;
        if (!wordDict.empty()) {
            minL = (int)wordDict[0].size();
            for (auto& w : wordDict) {
                int L = (int)w.size();
                minL = min(minL, L);
                maxL = max(maxL, L);
            }
        }
        vector<char> dp(n + 1, 0);
        dp[0] = 1;
        for (int i = 1; i <= n; ++i) {
            int L = max(1, minL);
            int R = maxL == 0 ? i : min(maxL, i);
            for (int len = L; len <= R; ++len) {
                int j = i - len;
                if (j >= 0 && dp[j]) {
                    if (dict.count(s.substr(j, len))) {
                        dp[i] = 1;
                        break;
                    }
                }
            }
        }
        return dp[n];
    }
};

static bool bfsCheck(const string& s, const vector<string>& wordDict) {
    unordered_set<string> dict(wordDict.begin(), wordDict.end());
    int n = (int)s.size();
    vector<char> vis(n + 1, 0);
    vector<int> q;
    q.push_back(0);
    vis[0] = 1;
    int head = 0;
    int minL = 1, maxL = 0;
    if (!wordDict.empty()) {
        minL = (int)wordDict[0].size();
        for (auto& w : wordDict) {
            int L = (int)w.size();
            minL = min(minL, L);
            maxL = max(maxL, L);
        }
    }
    while (head < (int)q.size()) {
        int i = q[head++];
        int L = max(1, minL);
        int R = maxL == 0 ? n - i : min(maxL, n - i);
        for (int len = L; len <= R; ++len) {
            int j = i + len;
            if (!vis[j] && dict.count(s.substr(i, len))) {
                if (j == n) return true;
                vis[j] = 1;
                q.push_back(j);
            }
        }
    }
    return n == 0;
}

int main() {
    Solution sol;
    {
        string s = "leetcode";
        vector<string> dict{"leet", "code"};
        assert(sol.wordBreak(s, dict) == true);
    }
    {
        string s = "applepenapple";
        vector<string> dict{"apple", "pen"};
        assert(sol.wordBreak(s, dict) == true);
    }
    {
        string s = "catsandog";
        vector<string> dict{"cats", "dog", "sand", "and", "cat"};
        assert(sol.wordBreak(s, dict) == false);
    }
    {
        string s = "";
        vector<string> dict{};
        assert(sol.wordBreak(s, dict) == true);
    }
    mt19937 rng(20260410);
    uniform_int_distribution<int> wordLenDist(1, 6);
    uniform_int_distribution<int> alphabet(0, 2);
    auto makeWord = [&](int len) {
        string w;
        w.resize(len);
        for (int i = 0; i < len; ++i) w[i] = char('a' + alphabet(rng));
        return w;
    };
    for (int tc = 0; tc < 500; ++tc) {
        int dictSize = 1 + (rng() % 8);
        vector<string> dict;
        dict.reserve(dictSize);
        unordered_set<string> seen;
        while ((int)dict.size() < dictSize) {
            string w = makeWord(wordLenDist(rng));
            if (seen.insert(w).second) dict.push_back(w);
        }
        int parts = 1 + (rng() % 8);
        string s;
        for (int i = 0; i < parts; ++i) s += dict[rng() % dict.size()];
        {
            string s2 = s;
            vector<string> d2 = dict;
            bool a = sol.wordBreak(s2, d2);
            bool b = bfsCheck(s, dict);
            assert(a == b);
        }
        {
            string s3 = s + "z";
            vector<string> d3 = dict;
            bool a = sol.wordBreak(s3, d3);
            assert(a == false);
        }
    }
    cout << "All tests passed." << "\n";
    return 0;
}

