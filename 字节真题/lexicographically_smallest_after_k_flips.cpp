#include <iostream>
#include <string>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, k;
    cin >> n >> k;
    string s;
    cin >> s;

    int used = 0;
    string ans;
    ans.reserve(n);

    for (int i = 0; i < n; ++i) {
        int rem = n - 1 - i;
        auto feasible = [&](int usedFlips) -> bool {
            int kp = k - usedFlips;
            if (kp < 0) return false;
            int needParity = kp & 1;
            int mx = kp < rem ? kp : rem;
            return mx >= needParity;
        };

        if (s[i] == '0') {
            if (feasible(used)) {
                ans.push_back('0');
            } else {
                ++used;
                ans.push_back('1');
            }
        } else {
            if (feasible(used + 1)) {
                ++used;
                ans.push_back('0');
            } else {
                ans.push_back('1');
            }
        }
    }

    cout << ans << "\n";
    return 0;
}

