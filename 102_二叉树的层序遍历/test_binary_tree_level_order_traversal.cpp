#include <cassert>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (!root) return res;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            int sz = (int)q.size();
            vector<int> level;
            level.reserve(sz);
            for (int i = 0; i < sz; ++i) {
                TreeNode* cur = q.front();
                q.pop();
                level.push_back(cur->val);
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
            }
            res.push_back(level);
        }
        return res;
    }
};

static void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

static vector<vector<int>> bruteLevelOrder(TreeNode* root) {
    vector<vector<int>> res;
    if (!root) return res;
    queue<pair<TreeNode*, int>> q;
    q.push({root, 0});
    while (!q.empty()) {
        auto [node, d] = q.front();
        q.pop();
        if ((int)res.size() == d) res.push_back({});
        res[d].push_back(node->val);
        if (node->left) q.push({node->left, d + 1});
        if (node->right) q.push({node->right, d + 1});
    }
    return res;
}

static TreeNode* genRandomTree(mt19937& rng, int n, int minVal, int maxVal) {
    if (n == 0) return nullptr;
    uniform_int_distribution<int> valDist(minVal, maxVal);
    vector<TreeNode*> nodes;
    nodes.reserve(n);
    for (int i = 0; i < n; ++i) nodes.push_back(new TreeNode(valDist(rng)));

    TreeNode* root = nodes[0];
    int idx = 1;
    for (int i = 0; i < n && idx < n; ++i) {
        if (idx < n) nodes[i]->left = nodes[idx++];
        if (idx < n) nodes[i]->right = nodes[idx++];
    }
    return root;
}

int main() {
    Solution sol;

    {
        TreeNode* root = new TreeNode(3);
        root->left = new TreeNode(9);
        root->right = new TreeNode(20);
        root->right->left = new TreeNode(15);
        root->right->right = new TreeNode(7);
        auto got = sol.levelOrder(root);
        vector<vector<int>> want{{3}, {9, 20}, {15, 7}};
        assert(got == want);
        freeTree(root);
    }
    {
        TreeNode* root = new TreeNode(1);
        auto got = sol.levelOrder(root);
        vector<vector<int>> want{{1}};
        assert(got == want);
        freeTree(root);
    }
    {
        TreeNode* root = nullptr;
        auto got = sol.levelOrder(root);
        vector<vector<int>> want{};
        assert(got == want);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 2000);
    for (int tc = 0; tc < 500; ++tc) {
        int n = nDist(rng);
        TreeNode* root = genRandomTree(rng, n, -1000, 1000);
        auto got = sol.levelOrder(root);
        auto want = bruteLevelOrder(root);
        assert(got == want);
        freeTree(root);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

