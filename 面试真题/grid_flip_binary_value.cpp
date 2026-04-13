#include <iostream>
#include <unordered_map>

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

static long long modInv(long long a) {
    a %= MOD;
    if (a < 0) a += MOD;
    return modPow(a, MOD - 2);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    long long n, m;
    int k;
    cin >> n >> m >> k;

    unordered_map<long long, char> rowParity;
    unordered_map<long long, char> colParity;
    rowParity.reserve((size_t)k * 2);
    colParity.reserve((size_t)k * 2);

    for (int i = 0; i < k; ++i) {
        int x;
        long long y;
        cin >> x >> y;
        if (x == 1) {
            colParity[y] ^= 1;
        } else {
            rowParity[y] ^= 1;
        }
    }

    long long inv2 = (MOD + 1) / 2;
    long long r = modPow(inv2, m);

    long long expN = (long long)((__int128)(n % (MOD - 1)) * (m % (MOD - 1)) % (MOD - 1));
    long long pow2N = modPow(2, expN);

    long long rn = modPow(r, n);
    long long sumRowsAll = pow2N * ((1 - rn + MOD) % MOD) % MOD * modInv((1 - r + MOD) % MOD) % MOD;

    long long sumPowRowsR = 0;
    for (auto& kv : rowParity) {
        if (kv.second & 1) {
            long long idx = kv.first;
            if (idx >= 1 && idx <= n) {
                sumPowRowsR += modPow(r, idx - 1);
                if (sumPowRowsR >= MOD) sumPowRowsR -= MOD;
            }
        }
    }
    long long sumRowsR = pow2N * sumPowRowsR % MOD;

    long long B = 0;
    for (auto& kv : colParity) {
        if (kv.second & 1) {
            long long idx = kv.first;
            if (idx >= 1 && idx <= m) {
                B += modPow(inv2, idx);
                if (B >= MOD) B -= MOD;
            }
        }
    }

    long long A = (1 - r + MOD) % MOD;
    long long S1 = sumRowsR * ((A - B + MOD) % MOD) % MOD;
    long long S2 = ((sumRowsAll - sumRowsR + MOD) % MOD) * B % MOD;
    long long ans = (S1 + S2) % MOD;

    cout << ans << "\n";
    return 0;
}

