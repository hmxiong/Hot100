#include <iostream>

using namespace std;

static const long long MOD = 1000000007LL;

static long long modPow(long long a, long long e) {
    long long r = 1 % MOD;
    a %= MOD;
    while (e > 0) {
        if (e & 1) r = (r * a) % MOD;
        a = (a * a) % MOD;
        e >>= 1;
    }
    return r;
} 

static long long pow2(long long e) {
    return modPow(2, e);
}

static long long g(long long m, long long x) {
    if (m == 0) return x % MOD;
    if (x == 2) return pow2(m + 1);
    long long y = x / 2 + 1;
    long long res = (2LL * g(m - 1, y)) % MOD;
    if (x & 1) res = (res + pow2(m)) % MOD;
    return res;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    cin >> T;
    while (T--) {
        long long n, m;
        cin >> n >> m;
        cout << g(m, n) % MOD << "\n";
    }
    return 0;
}

