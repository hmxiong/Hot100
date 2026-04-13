#include <iostream>
#include <stack>
#include <unordered_map>
#include <vector>

using namespace std;

// 题目描述：
// 小苯有一个长度为 n 的数组 {a1, a2, a3, ..., an}，如果一个数组有一个唯一的最大值，那么这个数组就是一个好数组。小红想知道这个数组有多少连续子数组是好数组。

// 连续子数组是指在原数组中，连续的选择一段元素组成的新数组。

// 数据范围

// 1 <= n <= 100000
// 1 <= ai <= 10^9
// 输入格式

// 第一行输入一个整数 n 表示数组中的元素数量。

// 第二行输入 n 个整数 a1, a2, ..., an 表示数组元素。

// 输出格式

// 在一行上输出一个整数，表示有多少连续子数组是好数组。

//解题思路：单调栈 + 贡献法，维护4个不同的数组
//prevGreater[i]：i左侧第一个大于a[i]的元素的索引
//nextGreater[i]：i右侧第一个大于a[i]的元素的索引
//prevEq[i]：i左侧第一个等于a[i]的元素的索引
//nextEq[i]：i右侧第一个等于a[i]的元素的索引
//整体贡献的计算：i左侧第一个大于a[i]的元素的索引为leftBlock，i右侧第一个大于a[i]的元素的索引为rightBlock
//rightChoices = rightBlock - i
//leftChoices = i - leftBlock
//ans += leftChoices * rightChoices

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; ++i) cin >> a[i];

    vector<int> prevGreater(n + 1, 0), nextGreater(n + 1, n + 1);
    {
        stack<int> st;
        for (int i = 1; i <= n; ++i) {
            while (!st.empty() && a[st.top()] <= a[i]) st.pop();
            prevGreater[i] = st.empty() ? 0 : st.top();
            st.push(i);
        }
    }
    {
        stack<int> st;
        for (int i = n; i >= 1; --i) {
            while (!st.empty() && a[st.top()] <= a[i]) st.pop();
            nextGreater[i] = st.empty() ? (n + 1) : st.top();
            st.push(i);
        }
    }

    vector<int> prevEq(n + 1, 0), nextEq(n + 1, n + 1);
    {
        unordered_map<long long, int> last;
        last.reserve((size_t)n * 2);
        last.max_load_factor(0.7f);
        for (int i = 1; i <= n; ++i) {
            auto it = last.find(a[i]);
            prevEq[i] = (it == last.end() ? 0 : it->second);
            last[a[i]] = i;
        }
    }
    {
        unordered_map<long long, int> nxt;
        nxt.reserve((size_t)n * 2);
        nxt.max_load_factor(0.7f);
        for (int i = n; i >= 1; --i) {
            auto it = nxt.find(a[i]);
            nextEq[i] = (it == nxt.end() ? (n + 1) : it->second);
            nxt[a[i]] = i;
        }
    }

    long long ans = 0;
    for (int i = 1; i <= n; ++i) {
        int leftBlock = prevGreater[i];
        if (prevEq[i] > leftBlock) leftBlock = prevEq[i];
        int rightBlock = nextGreater[i];
        if (nextEq[i] < rightBlock) rightBlock = nextEq[i];
        long long leftChoices = i - leftBlock;
        long long rightChoices = rightBlock - i;
        ans += leftChoices * rightChoices;
    }

    cout << ans << "\n";
    return 0;
}

