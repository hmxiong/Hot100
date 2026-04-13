#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    string s;
    cin >> s;
    vector<long long> w(n);
    for (int i = 1; i <= n - 1; ++i) cin >> w[i];

    long long base = 0;
    long long best1 = numeric_limits<long long>::min();
    long long best2 = numeric_limits<long long>::min();

    for (int i = 1; i <= n - 1; ++i) {
        bool eq = (s[i - 1] == s[i]);
        if (eq) base += w[i];
        long long d = eq ? -w[i] : w[i];
        if (d > best1) {
            best2 = best1;
            best1 = d;
        } else if (d > best2) {
            best2 = d;
        }
    }

    long long bestDelta = 0;
    if (best1 > bestDelta) bestDelta = best1;
    if (n - 1 >= 2 && best1 != numeric_limits<long long>::min() && best2 != numeric_limits<long long>::min()) {
        if (best1 + best2 > bestDelta) bestDelta = best1 + best2;
    }

    cout << (base + bestDelta) << "\n";
    return 0;
}

