#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = (int)temperatures.size();
        vector<int> ans(n, 0);
        vector<int> st;
        st.reserve(n);

        for (int i = 0; i < n; ++i) {
            while (!st.empty() && temperatures[i] > temperatures[st.back()]) {
                int j = st.back();
                st.pop_back();
                ans[j] = i - j;
            }
            st.push_back(i);
        }
        return ans;
    }
};

static vector<int> bruteDailyTemperatures(const vector<int>& t) {
    int n = (int)t.size();
    vector<int> ans(n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (t[j] > t[i]) {
                ans[i] = j - i;
                break;
            }
        }
    }
    return ans;
}

static void expectEq(const vector<int>& a, const vector<int>& b) {
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) assert(a[i] == b[i]);
}

int main() {
    Solution sol;

    {
        vector<int> t{73, 74, 75, 71, 69, 72, 76, 73};
        auto got = sol.dailyTemperatures(t);
        expectEq(got, vector<int>({1, 1, 4, 2, 1, 1, 0, 0}));
    }
    {
        vector<int> t{30, 40, 50, 60};
        auto got = sol.dailyTemperatures(t);
        expectEq(got, vector<int>({1, 1, 1, 0}));
    }
    {
        vector<int> t{30, 60, 90};
        auto got = sol.dailyTemperatures(t);
        expectEq(got, vector<int>({1, 1, 0}));
    }
    {
        vector<int> t{80};
        auto got = sol.dailyTemperatures(t);
        expectEq(got, vector<int>({0}));
    }
    {
        vector<int> t{100, 99, 98, 97};
        auto got = sol.dailyTemperatures(t);
        expectEq(got, vector<int>({0, 0, 0, 0}));
    }
    {
        vector<int> t{30, 30, 30, 31};
        auto got = sol.dailyTemperatures(t);
        expectEq(got, vector<int>({3, 2, 1, 0}));
    }

    mt19937 rng(20260409);
    uniform_int_distribution<int> nDist(1, 400);
    uniform_int_distribution<int> tDist(30, 100);
    for (int tc = 0; tc < 5000; ++tc) {
        int n = nDist(rng);
        vector<int> t(n);
        for (int i = 0; i < n; ++i) t[i] = tDist(rng);

        vector<int> copy = t;
        auto got = sol.dailyTemperatures(copy);
        auto want = bruteDailyTemperatures(t);
        expectEq(got, want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

