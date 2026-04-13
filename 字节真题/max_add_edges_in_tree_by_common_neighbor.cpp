#include <iostream>
#include <vector>

using namespace std;

// 小红有一棵 n 个点，n-1 条边的树。如果树上存在一个点 w，使得原始的树上存在边 (u, w) 和 (w, v)，那么我们可以添加一条边 (u, v)。小红想知道她最多可以添加多少条边。

// 树是指这样的一张图，其上的任意两个点都连通，且不存在环。

//解题思路：贡献法，维护每个点的度，每个点的贡献为 C(deg[i], 2)，表示可以添加的边数
//最后将所有点的贡献累加起来，就是答案

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    vector<long long> deg(n + 1, 0);
    for (int i = 0; i < n - 1; ++i) {
        int u, v;
        cin >> u >> v;
        ++deg[u];
        ++deg[v];
    }

    long long ans = 0;
    for (int i = 1; i <= n; ++i) {
        ans += deg[i] * (deg[i] - 1) / 2; //直接根据度来计算 C(deg[i], 2)，表示可以添加的边数
    }

    cout << ans << "\n";
    return 0;
}

