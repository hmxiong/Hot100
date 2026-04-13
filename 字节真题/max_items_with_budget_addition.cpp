#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;


// 小苯有 n 个钱包，其中第 i 个钱包装了 wallets[i] 元，他每天都会恰好使用一个钱包中的钱去购物，尽可能多地购买一种单价为 k 元的物品，日复一日，直到所有钱包中的钱都分别买不起此物品。

// 在小苯开始日复一日地购物前，可以向任意一些钱包中再加入一些钱，但总共加入的钱数不超过 m。

// 现在小苯想知道，如果自己以最优方案向钱包中加钱，那么最多可以购买多少件此物品。

// 解题思路：

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    long long k, m;
    cin >> n >> k >> m;
    vector<long long> wallets(n);
    for (int i = 0; i < n; ++i) cin >> wallets[i];

    long long base = 0;
    vector<long long> costs;
    costs.reserve(n);

    for (int i = 0; i < n; ++i) {
        base += wallets[i] / k;
        long long r = wallets[i] % k;
        if (r != 0) costs.push_back(k - r);
    }

    sort(costs.begin(), costs.end());

    long long extra = 0;
    for (long long c : costs) {
        if (m < c) break;
        m -= c;
        ++extra;
    }
    extra += m / k;

    cout << (base + extra) << "\n";
    return 0;
}

