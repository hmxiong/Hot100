#include <cassert>
#include <climits>
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
    bool isValidBST(TreeNode* root) {
        return dfs(root, LLONG_MIN, LLONG_MAX);
    }

private:
    static bool dfs(TreeNode* node, long long lo, long long hi) {
        if (!node) return true;
        long long v = node->val;
        if (!(lo < v && v < hi)) return false;
        return dfs(node->left, lo, v) && dfs(node->right, v, hi);
    }
};

static void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

static void inorder(TreeNode* root, vector<long long>& out) {
    if (!root) return;
    inorder(root->left, out);
    out.push_back(root->val);
    inorder(root->right, out);
}

static bool bruteIsValidBST(TreeNode* root) {
    vector<long long> seq;
    inorder(root, seq);
    for (int i = 1; i < (int)seq.size(); ++i) {
        if (seq[i] <= seq[i - 1]) return false;
    }
    return true;
}

static TreeNode* genRandomTree(mt19937& rng, int n) {
    if (n == 0) return nullptr;
    uniform_int_distribution<int> valDist(INT_MIN, INT_MAX);
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
        TreeNode* root = new TreeNode(2);
        root->left = new TreeNode(1);
        root->right = new TreeNode(3);
        assert(sol.isValidBST(root) == true);
        freeTree(root);
    }
    {
        TreeNode* root = new TreeNode(5);
        root->left = new TreeNode(1);
        root->right = new TreeNode(4);
        root->right->left = new TreeNode(3);
        root->right->right = new TreeNode(6);
        assert(sol.isValidBST(root) == false);
        freeTree(root);
    }
    {
        TreeNode* root = new TreeNode(INT_MIN);
        root->right = new TreeNode(INT_MAX);
        assert(sol.isValidBST(root) == true);
        freeTree(root);
    }
    {
        TreeNode* root = new TreeNode(1);
        root->left = new TreeNode(1);
        assert(sol.isValidBST(root) == false);
        freeTree(root);
    }
    {
        TreeNode* root = new TreeNode(10);
        root->left = new TreeNode(5);
        root->right = new TreeNode(15);
        root->right->left = new TreeNode(6);
        root->right->right = new TreeNode(20);
        assert(sol.isValidBST(root) == false);
        freeTree(root);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 2000);
    for (int tc = 0; tc < 500; ++tc) {
        int n = nDist(rng);
        TreeNode* root = genRandomTree(rng, n);
        bool got = sol.isValidBST(root);
        bool want = bruteIsValidBST(root);
        assert(got == want);
        freeTree(root);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

