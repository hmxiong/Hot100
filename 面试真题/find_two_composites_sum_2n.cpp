#include <iostream>
#include <vector>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T;
    cin >> T;
    vector<int> ns(T);
    int maxN = 0;
    for (int i = 0; i < T; ++i) {
        cin >> ns[i];
        if (ns[i] > maxN) maxN = ns[i];
    }

    int limit = 2 * maxN;
    vector<char> isPrime(limit + 1, 1);
    if (limit >= 0) isPrime[0] = 0;
    if (limit >= 1) isPrime[1] = 0;
    for (long long i = 2; i * i <= limit; ++i) {
        if (!isPrime[(int)i]) continue;
        for (long long j = i * i; j <= limit; j += i) isPrime[(int)j] = 0;
    }
    auto isComposite = [&](int x) -> bool {
        if (x < 4) return false;
        return !isPrime[x];
    };

    const int candidates[] = {6, 4, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30};

    for (int idx = 0; idx < T; ++idx) {
        int n = ns[idx];
        int total = 2 * n;
        if (total < 8) {
            cout << -1 << "\n";
            continue;
        }
        bool found = false;
        for (int x : candidates) {
            int y = total - x;
            if (y < 4) continue;
            if (isComposite(x) && isComposite(y)) {
                if (x < y) {
                    cout << x << " " << y << "\n";
                } else {
                    cout << y << " " << x << "\n";
                }
                found = true;
                break;
            }
        }
        if (!found) cout << -1 << "\n";
    }
    return 0;
}
