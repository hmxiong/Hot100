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
    vector<int> rightSideView(TreeNode* root) {
        vector<int> ans;
        if (!root) return ans;

        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            int sz = (int)q.size();
            for (int i = 0; i < sz; ++i) {
                TreeNode* cur = q.front();
                q.pop();
                if (cur->left) q.push(cur->left);
                if (cur->right) q.push(cur->right);
                if (i == sz - 1) ans.push_back(cur->val);
            }
        }
        return ans;
    }
};

static void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

static void dfsRightFirst(TreeNode* node, int depth, vector<int>& out) {
    if (!node) return;
    if ((int)out.size() == depth) out.push_back(node->val);
    dfsRightFirst(node->right, depth + 1, out);
    dfsRightFirst(node->left, depth + 1, out);
}

static vector<int> refRightView(TreeNode* root) {
    vector<int> out;
    dfsRightFirst(root, 0, out);
    return out;
}

static TreeNode* genRandomTree(mt19937& rng, int n) {
    if (n == 0) return nullptr;
    uniform_int_distribution<int> valDist(-100, 100);
    vector<TreeNode*> nodes;
    nodes.reserve(n);
    for (int i = 0; i < n; ++i) nodes.push_back(new TreeNode(valDist(rng)));

    TreeNode* root = nodes[0];
    uniform_real_distribution<double> prob(0.0, 1.0);
    int idx = 1;
    for (int i = 0; i < n && idx < n; ++i) {
        if (idx < n && prob(rng) < 0.85) nodes[i]->left = nodes[idx++];
        if (idx < n && prob(rng) < 0.85) nodes[i]->right = nodes[idx++];
    }
    return root;
}

static void expectEq(const vector<int>& a, const vector<int>& b) {
    assert(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) assert(a[i] == b[i]);
}

int main() {
    Solution sol;

    {
        TreeNode* root = new TreeNode(1);
        root->left = new TreeNode(2);
        root->right = new TreeNode(3);
        root->left->right = new TreeNode(5);
        root->right->right = new TreeNode(4);
        vector<int> got = sol.rightSideView(root);
        vector<int> want{1, 3, 4};
        expectEq(got, want);
        freeTree(root);
    }
    {
        TreeNode* root = new TreeNode(1);
        root->left = new TreeNode(2);
        root->right = new TreeNode(3);
        root->left->left = new TreeNode(4);
        root->left->left->left = new TreeNode(5);
        vector<int> got = sol.rightSideView(root);
        vector<int> want{1, 3, 4, 5};
        expectEq(got, want);
        freeTree(root);
    }
    {
        TreeNode* root = new TreeNode(1);
        root->right = new TreeNode(3);
        vector<int> got = sol.rightSideView(root);
        vector<int> want{1, 3};
        expectEq(got, want);
        freeTree(root);
    }
    {
        TreeNode* root = nullptr;
        vector<int> got = sol.rightSideView(root);
        vector<int> want;
        expectEq(got, want);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 100);
    for (int tc = 0; tc < 2000; ++tc) {
        int n = nDist(rng);
        TreeNode* root = genRandomTree(rng, n);
        vector<int> got = sol.rightSideView(root);
        vector<int> want = refRightView(root);
        expectEq(got, want);
        freeTree(root);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

