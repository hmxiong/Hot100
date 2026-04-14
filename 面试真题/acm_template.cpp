/*
 * 常用编译指令: g++ -O2 -std=c++17 acm_template.cpp -o acm_template
 * 运行指令: ./acm_template
 * 快速测试指令 (从文件读入): ./acm_template < input.txt
 */
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <set>
#include <queue>
#include <stack>
#include <cmath>

using namespace std;

class Solution {
public:
    // 在这里实现你的核心逻辑
    void solve() {
        // TODO: 补充具体的解题代码
        
    }
};

int main() {
    // 优化标准I/O流性能，ACM模式下必备
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);

    // ==========================================
    // 模式1：多组测试数据，第一行给定了组数 t
    // ==========================================
    /*
    int t;
    if (cin >> t) {
        while (t--) {
            Solution sol;
            sol.solve();
        }
    }
    */

    // ==========================================
    // 模式2：持续读取直到 EOF (遇到文件尾或Ctrl+D)
    // ==========================================
    /*
    int n;
    while (cin >> n) {
        // 读取其它参数
        Solution sol;
        sol.solve();
    }
    */

    // ==========================================
    // 模式3：单组测试数据（默认打开）
    // ==========================================
    Solution sol;
    sol.solve();

    return 0;
}
