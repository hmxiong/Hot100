#include <array>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

static int bitOf(char c) {
    if (c == 'r') return 0;
    if (c == 'e') return 1;
    return 2;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, q;
    cin >> n >> q;
    string s;
    cin >> s;

    vector<array<int, 8>> prefixCount(n + 1);
    array<int, 8> cur{};
    cur.fill(0);

    int mask = 0;
    cur[mask] += 1;
    prefixCount[0] = cur;

    for (int i = 1; i <= n; ++i) {
        mask ^= (1 << bitOf(s[i - 1]));
        cur[mask] += 1;
        prefixCount[i] = cur;
    }

    while (q--) {
        int l, r;
        cin >> l >> r;
        int left = l - 1;
        int right = r;

        long long cnt[8];
        for (int m = 0; m < 8; ++m) {
            int a = prefixCount[right][m];
            int b = (left > 0 ? prefixCount[left - 1][m] : 0);
            cnt[m] = (long long)(a - b);
        }

        long long ans = 0;
        for (int m = 0; m < 8; ++m) ans += cnt[m] * (cnt[m] - 1) / 2;

        for (int bit = 0; bit < 3; ++bit) {
            int d = (1 << bit);
            for (int m = 0; m < 8; ++m) {
                int t = m ^ d;
                if (m < t) ans += cnt[m] * cnt[t];
            }
        }

        cout << ans << "\n";
    }

    return 0;
}
