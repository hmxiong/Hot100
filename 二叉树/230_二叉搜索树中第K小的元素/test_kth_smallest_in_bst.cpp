#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <unordered_set>
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
    int kthSmallest(TreeNode* root, int k) {
        vector<TreeNode*> st;
        TreeNode* cur = root;
        while (cur || !st.empty()) {
            while (cur) {
                st.push_back(cur);
                cur = cur->left;
            }
            cur = st.back();
            st.pop_back();
            k--;
            if (k == 0) return cur->val;
            cur = cur->right;
        }
        return -1;
    }
};

static void freeTree(TreeNode* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    delete root;
}

static void insertBST(TreeNode*& root, int v) {
    if (!root) {
        root = new TreeNode(v);
        return;
    }
    TreeNode* cur = root;
    while (true) {
        if (v < cur->val) {
            if (!cur->left) {
                cur->left = new TreeNode(v);
                return;
            }
            cur = cur->left;
        } else if (v > cur->val) {
            if (!cur->right) {
                cur->right = new TreeNode(v);
                return;
            }
            cur = cur->right;
        } else {
            return;
        }
    }
}

static vector<int> collectValues(TreeNode* root) {
    vector<int> v;
    vector<TreeNode*> st;
    TreeNode* cur = root;
    while (cur || !st.empty()) {
        while (cur) {
            st.push_back(cur);
            cur = cur->left;
        }
        cur = st.back();
        st.pop_back();
        v.push_back(cur->val);
        cur = cur->right;
    }
    return v;
}

int main() {
    Solution sol;

    {
        TreeNode* root = nullptr;
        for (int x : vector<int>{3, 1, 4, 2}) insertBST(root, x);
        assert(sol.kthSmallest(root, 1) == 1);
        freeTree(root);
    }
    {
        TreeNode* root = nullptr;
        for (int x : vector<int>{5, 3, 6, 2, 4, 1}) insertBST(root, x);
        assert(sol.kthSmallest(root, 3) == 3);
        freeTree(root);
    }
    {
        TreeNode* root = nullptr;
        insertBST(root, 42);
        assert(sol.kthSmallest(root, 1) == 42);
        freeTree(root);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(1, 2000);
    uniform_int_distribution<int> valDist(0, 100000);

    for (int tc = 0; tc < 1000; ++tc) {
        int n = nDist(rng);
        TreeNode* root = nullptr;
        unordered_set<int> used;
        used.reserve((size_t)n * 2);
        while ((int)used.size() < n) {
            int x = valDist(rng);
            if (used.insert(x).second) insertBST(root, x);
        }

        vector<int> sorted = collectValues(root);
        vector<int> sorted2 = sorted;
        sort(sorted2.begin(), sorted2.end());
        assert(sorted == sorted2);

        uniform_int_distribution<int> kDist(1, n);
        for (int q = 0; q < 20; ++q) {
            int k = kDist(rng);
            int got = sol.kthSmallest(root, k);
            int want = sorted2[k - 1];
            assert(got == want);
        }
        freeTree(root);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

