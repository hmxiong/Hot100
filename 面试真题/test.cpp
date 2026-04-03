#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
using namespace std;

static vector<int> sieve_primes(int limit) {
    vector<uint8_t> bs(limit + 1, 1);
    bs[0] = bs[1] = 0;
    int r = (int)floor(sqrt(limit));
    for (int i = 2; i <= r; ++i) {
        if (bs[i]) {
            for (long long j = 1LL * i * i; j <= limit; j += i) bs[(int)j] = 0;
        }
    }
    vector<int> ps;
    for (int i = 2; i <= limit; ++i) if (bs[i]) ps.push_back(i);
    return ps;
}

static const vector<int> PRIMES = sieve_primes(31623);

static vector<pair<int,int>> factorize(int x) {
    vector<pair<int,int>> res;
    int n = x;
    for (int p : PRIMES) {
        if (1LL * p * p > n) break;
        if (n % p == 0) {
            int c = 0;
            while (n % p == 0) { n /= p; ++c; }
            res.emplace_back(p, c);
        }
    }
    if (n > 1) res.emplace_back(n, 1);
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int T;
    if (!(cin >> T)) return 0;
    vector<string> outs;
    outs.reserve(T);
    while (T--) {
        int n;
        cin >> n;
        vector<int> a(n);
        for (int i = 0; i < n; ++i) cin >> a[i];
        vector<vector<pair<int,int>>> facts(n);
        unordered_map<int,int> gmax;
        gmax.reserve(n * 2);
        for (int i = 0; i < n; ++i) {
            facts[i] = factorize(a[i]);
            for (auto &pr : facts[i]) {
                int p = pr.first, c = pr.second;
                auto it = gmax.find(p);
                if (it == gmax.end() || it->second < c) gmax[p] = c;
            }
        }
        vector<int> targets;
        targets.reserve(gmax.size());
        for (auto &kv : gmax) if (kv.second > 0) targets.push_back(kv.first);
        if (targets.empty()) {
            outs.emplace_back("1");
            continue;
        }
        vector<vector<int>> coverage(n);
        for (int i = 0; i < n; ++i) {
            for (auto &pr : facts[i]) {
                int p = pr.first, c = pr.second;
                auto it = gmax.find(p);
                if (it != gmax.end() && it->second == c) coverage[i].push_back(p);
            }
        }
        int need = (int)targets.size();
        unordered_map<int,int> cnt;
        cnt.reserve(need * 2);
        int have = 0;
        int ans = n;
        int l = 0;
        for (int r = 0; r < n; ++r) {
            for (int p : coverage[r]) {
                int c = ++cnt[p];
                if (c == 1) ++have;
            }
            while (have == need && l <= r) {
                ans = min(ans, r - l + 1);
                for (int p : coverage[l]) {
                    int c = --cnt[p];
                    if (c == 0) --have;
                }
                ++l;
            }
        }
        outs.emplace_back(to_string(ans));
    }
    for (size_t i = 0; i < outs.size(); ++i) {
        if (i) cout << '\n';
        cout << outs[i];
    }
    return 0;
}
