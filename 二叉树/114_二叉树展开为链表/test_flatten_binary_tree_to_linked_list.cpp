#include <cassert>
#include <iostream>
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
    void flatten(TreeNode* root) {
        TreeNode* cur = root;
        while (cur) {
            if (cur->left) {
                TreeNode* pred = cur->left;
                while (pred->right) pred = pred->right;
                pred->right = cur->right;
                cur->right = cur->left;
                cur->left = nullptr;
            }
            cur = cur->right;
        }
    }
};

static void preorder(TreeNode* root, vector<int>& out) {
    if (!root) return;
    out.push_back(root->val);
    preorder(root->left, out);
    preorder(root->right, out);
}

static vector<int> rightChain(TreeNode* root) {
    vector<int> out;
    TreeNode* cur = root;
    while (cur) {
        assert(cur->left == nullptr);
        out.push_back(cur->val);
        cur = cur->right;
    }
    return out;
}

static void freeFlattened(TreeNode* root) {
    TreeNode* cur = root;
    while (cur) {
        TreeNode* nxt = cur->right;
        delete cur;
        cur = nxt;
    }
}

static TreeNode* buildExample1() {
    TreeNode* root = new TreeNode(1);
    root->left = new TreeNode(2);
    root->right = new TreeNode(5);
    root->left->left = new TreeNode(3);
    root->left->right = new TreeNode(4);
    root->right->right = new TreeNode(6);
    return root;
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
        TreeNode* root = buildExample1();
        vector<int> want;
        preorder(root, want);
        sol.flatten(root);
        vector<int> got = rightChain(root);
        assert(got == want);
        freeFlattened(root);
    }
    {
        TreeNode* root = nullptr;
        sol.flatten(root);
        assert(root == nullptr);
    }
    {
        TreeNode* root = new TreeNode(0);
        vector<int> want;
        preorder(root, want);
        sol.flatten(root);
        vector<int> got = rightChain(root);
        assert(got == want);
        freeFlattened(root);
    }

    mt19937 rng(20260409);
    uniform_int_distribution<int> nDist(0, 2000);
    for (int tc = 0; tc < 500; ++tc) {
        int n = nDist(rng);
        TreeNode* root = genRandomTree(rng, n, -100, 100);
        vector<int> want;
        preorder(root, want);
        sol.flatten(root);
        vector<int> got = rightChain(root);
        assert(got == want);
        freeFlattened(root);
    }

    cout << "All tests passed." << "\n";
    return 0;
}
