#include <cassert>
#include <cctype>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace std;

class Solution {
public:
    string decodeString(const string& s) {
        vector<int> cntSt;
        vector<string> strSt;
        int curNum = 0;
        string cur;

        for (char ch : s) {
            if (isdigit((unsigned char)ch)) {
                curNum = curNum * 10 + (ch - '0');
            } else if (ch == '[') {
                cntSt.push_back(curNum);
                strSt.push_back(cur);
                curNum = 0;
                cur.clear();
            } else if (ch == ']') {
                int k = cntSt.back();
                cntSt.pop_back();
                string prev = std::move(strSt.back());
                strSt.pop_back();
                string expanded;
                expanded.reserve(prev.size() + (size_t)k * cur.size());
                expanded += prev;
                for (int i = 0; i < k; ++i) expanded += cur;
                cur = std::move(expanded);
            } else {
                cur.push_back(ch);
            }
        }
        return cur;
    }
};

static string decodeRec(const string& s, int& i) {
    string out;
    int n = (int)s.size();
    while (i < n && s[i] != ']') {
        if (isdigit((unsigned char)s[i])) {
            int k = 0;
            while (i < n && isdigit((unsigned char)s[i])) {
                k = k * 10 + (s[i] - '0');
                i++;
            }
            i++; // skip '['
            string inner = decodeRec(s, i);
            i++; // skip ']'
            for (int t = 0; t < k; ++t) out += inner;
        } else {
            out.push_back(s[i]);
            i++;
        }
    }
    return out;
}

static string bruteDecode(const string& s) {
    int i = 0;
    return decodeRec(s, i);
}

static string randomLetters(mt19937& rng, int len) {
    uniform_int_distribution<int> d(0, 25);
    string s;
    s.reserve((size_t)len);
    for (int i = 0; i < len; ++i) s.push_back((char)('a' + d(rng)));
    return s;
}

static string genEncoded(mt19937& rng, int depth, int& remainingBudget) {
    if (remainingBudget <= 0) return "";
    uniform_int_distribution<int> partDist(1, 3);
    int parts = partDist(rng);
    string res;
    for (int p = 0; p < parts && remainingBudget > 0; ++p) {
        uniform_int_distribution<int> kindDist(0, depth > 0 ? 1 : 0);
        int kind = kindDist(rng); // 0 letters, 1 repeat-block
        if (kind == 0) {
            uniform_int_distribution<int> lenDist(1, 5);
            int len = min(lenDist(rng), remainingBudget);
            res += randomLetters(rng, len);
            remainingBudget -= len;
        } else {
            uniform_int_distribution<int> kDist(1, 10);
            int k = kDist(rng);
            string kStr = to_string(k);
            if (remainingBudget < (int)kStr.size() + 2) break;
            remainingBudget -= (int)kStr.size() + 2;
            res += kStr;
            res += '[';
            string inner = genEncoded(rng, depth - 1, remainingBudget);
            res += inner;
            res += ']';
        }
    }
    return res;
}

static void expectEq(const string& a, const string& b) {
    assert(a.size() == b.size());
    assert(a == b);
}

int main() {
    Solution sol;

    expectEq(sol.decodeString("3[a]2[bc]"), "aaabcbc");
    expectEq(sol.decodeString("3[a2[c]]"), "accaccacc");
    expectEq(sol.decodeString("2[abc]3[cd]ef"), "abcabccdcdcdef");
    expectEq(sol.decodeString("abc3[cd]xyz"), "abccdcdcdxyz");
    expectEq(sol.decodeString("10[a]"), "aaaaaaaaaa");
    expectEq(sol.decodeString("1[a]"), "a");
    expectEq(sol.decodeString("2[a]3[b]"), "aabbb");
    expectEq(sol.decodeString("3[a]2[b4[F]c]"), "aaabFFFFcbFFFFc");

    mt19937 rng(20260401);
    for (int tc = 0; tc < 5000; ++tc) {
        int budget = 30;
        string s = genEncoded(rng, 3, budget);
        string got = sol.decodeString(s);
        string want = bruteDecode(s);
        expectEq(got, want);
        assert(got.size() <= 100000u);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

