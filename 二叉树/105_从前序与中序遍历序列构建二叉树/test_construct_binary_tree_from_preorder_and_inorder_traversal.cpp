#include <cassert>
#include <iostream>
#include <random>
#include <unordered_map>
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
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = (int)preorder.size();
        unordered_map<int, int> inPos;
        inPos.reserve((size_t)n * 2);
        for (int i = 0; i < n; ++i) inPos[inorder[i]] = i;
        int preIdx = 0;
        return build(preorder, 0, n - 1, inPos, preIdx);
    }

private:
    static TreeNode* build(const vector<int>& preorder,
                           int inL,
                           int inR,
                           const unordered_map<int, int>& inPos,
                           int& preIdx) {
        if (inL > inR) return nullptr;
        int rootVal = preorder[preIdx++];
        TreeNode* root = new TreeNode(rootVal);
        int mid = inPos.at(rootVal);
        root->left = build(preorder, inL, mid - 1, inPos, preIdx);
        root->right = build(preorder, mid + 1, inR, inPos, preIdx);
        return root;
    }
};

static void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

static void preorderTrav(TreeNode* root, vector<int>& out) {
    if (!root) return;
    out.push_back(root->val);
    preorderTrav(root->left, out);
    preorderTrav(root->right, out);
}

static void inorderTrav(TreeNode* root, vector<int>& out) {
    if (!root) return;
    inorderTrav(root->left, out);
    out.push_back(root->val);
    inorderTrav(root->right, out);
}

static TreeNode* genRandomTree(mt19937& rng, int n) {
    if (n == 0) return nullptr;
    vector<int> vals(n);
    for (int i = 0; i < n; ++i) vals[i] = i + 1;
    shuffle(vals.begin(), vals.end(), rng);

    vector<TreeNode*> nodes;
    nodes.reserve(n);
    for (int i = 0; i < n; ++i) nodes.push_back(new TreeNode(vals[i]));

    TreeNode* root = nodes[0];
    vector<TreeNode*> candidates;
    candidates.reserve(n);
    candidates.push_back(root);

    int idx = 1;
    uniform_int_distribution<int> coin(0, 1);
    while (idx < n) {
        int pIdx = uniform_int_distribution<int>(0, (int)candidates.size() - 1)(rng);
        TreeNode* p = candidates[pIdx];

        bool attachLeftFirst = (coin(rng) == 0);
        auto tryAttach = [&](bool leftSide) -> bool {
            if (leftSide) {
                if (p->left) return false;
                p->left = nodes[idx++];
                candidates.push_back(p->left);
                return true;
            } else {
                if (p->right) return false;
                p->right = nodes[idx++];
                candidates.push_back(p->right);
                return true;
            }
        };

        if (!tryAttach(attachLeftFirst)) {
            if (!tryAttach(!attachLeftFirst)) {
                candidates[pIdx] = candidates.back();
                candidates.pop_back();
            }
        }
    }

    return root;
}

int main() {
    Solution sol;

    {
        vector<int> preorder{3, 9, 20, 15, 7};
        vector<int> inorder{9, 3, 15, 20, 7};
        TreeNode* root = sol.buildTree(preorder, inorder);
        vector<int> gotPre;
        vector<int> gotIn;
        preorderTrav(root, gotPre);
        inorderTrav(root, gotIn);
        assert(gotPre == preorder);
        assert(gotIn == inorder);
        freeTree(root);
    }
    {
        vector<int> preorder{-1};
        vector<int> inorder{-1};
        TreeNode* root = sol.buildTree(preorder, inorder);
        vector<int> gotPre;
        vector<int> gotIn;
        preorderTrav(root, gotPre);
        inorderTrav(root, gotIn);
        assert(gotPre == preorder);
        assert(gotIn == inorder);
        freeTree(root);
    }

    mt19937 rng(20260410);
    uniform_int_distribution<int> nDist(1, 300);
    for (int tc = 0; tc < 1000; ++tc) {
        int n = nDist(rng);
        TreeNode* root = genRandomTree(rng, n);
        vector<int> pre;
        vector<int> in;
        preorderTrav(root, pre);
        inorderTrav(root, in);

        vector<int> preCopy = pre;
        vector<int> inCopy = in;
        TreeNode* rebuilt = sol.buildTree(preCopy, inCopy);
        vector<int> gotPre;
        vector<int> gotIn;
        preorderTrav(rebuilt, gotPre);
        inorderTrav(rebuilt, gotIn);
        assert(gotPre == pre);
        assert(gotIn == in);

        freeTree(root);
        freeTree(rebuilt);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

