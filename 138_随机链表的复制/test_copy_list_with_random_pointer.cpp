#include <cassert>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

using namespace std;

struct Node {
    int val;
    Node* next;
    Node* random;
    Node(int x) : val(x), next(nullptr), random(nullptr) {}
};

class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;
        Node* cur = head;
        while (cur) {
            Node* cp = new Node(cur->val);
            cp->next = cur->next;
            cur->next = cp;
            cur = cp->next;
        }
        cur = head;
        while (cur) {
            Node* cp = cur->next;
            cp->random = cur->random ? cur->random->next : nullptr;
            cur = cp->next;
        }
        Node* newHead = head->next;
        Node* c1 = head;
        Node* c2 = newHead;
        while (c1) {
            c1->next = c2->next;
            c1 = c1->next;
            c2->next = c1 ? c1->next : nullptr;
            c2 = c2->next;
        }
        return newHead;
    }
};

static Node* buildList(const vector<int>& vals) {
    Node dummy(0);
    Node* tail = &dummy;
    for (int x : vals) {
        tail->next = new Node(x);
        tail = tail->next;
    }
    return dummy.next;
}

static int length(Node* head) {
    int n = 0;
    for (Node* p = head; p; p = p->next) n++;
    return n;
}

static void assignRandom(Node* head, const vector<int>& idx) {
    vector<Node*> nodes;
    for (Node* p = head; p; p = p->next) nodes.push_back(p);
    int n = (int)nodes.size();
    for (int i = 0; i < n; ++i) {
        int t = idx[i];
        if (t >= 0) nodes[i]->random = nodes[t];
        else nodes[i]->random = nullptr;
    }
}

static pair<vector<int>, vector<int>> toVectors(Node* head) {
    vector<Node*> nodes;
    for (Node* p = head; p; p = p->next) nodes.push_back(p);
    unordered_map<Node*, int> pos;
    for (int i = 0; i < (int)nodes.size(); ++i) pos[nodes[i]] = i;
    vector<int> vals;
    vector<int> rnd;
    for (Node* p : nodes) {
        vals.push_back(p->val);
        if (p->random) rnd.push_back(pos[p->random]);
        else rnd.push_back(-1);
    }
    return {vals, rnd};
}

static Node* bruteCopy(Node* head) {
    if (!head) return nullptr;
    unordered_map<Node*, Node*> mp;
    for (Node* p = head; p; p = p->next) mp[p] = new Node(p->val);
    for (Node* p = head; p; p = p->next) {
        mp[p]->next = p->next ? mp[p->next] : nullptr;
        mp[p]->random = p->random ? mp[p->random] : nullptr;
    }
    return mp[head];
}

static void freeList(Node* head) {
    while (head) {
        Node* nxt = head->next;
        delete head;
        head = nxt;
    }
}

int main() {
    Solution sol;
    {
        Node* a = buildList({7, 13, 11, 10, 1});
        assignRandom(a, {-1, 0, 4, 2, 0});
        Node* b = sol.copyRandomList(a);
        auto A = toVectors(a);
        auto B = toVectors(b);
        assert(A.first == B.first);
        assert(A.second == B.second);
        freeList(a);
        freeList(b);
    }
    {
        Node* a = nullptr;
        Node* b = sol.copyRandomList(a);
        assert(b == nullptr);
        freeList(b);
    }
    {
        Node* a = buildList({1});
        assignRandom(a, {0});
        Node* b = sol.copyRandomList(a);
        auto A = toVectors(a);
        auto B = toVectors(b);
        assert(A.first == B.first);
        assert(A.second == B.second);
        freeList(a);
        freeList(b);
    }
    {
        Node* a = buildList({1, 2});
        assignRandom(a, {-1, 0});
        Node* b = sol.copyRandomList(a);
        auto A = toVectors(a);
        auto B = toVectors(b);
        assert(A.first == B.first);
        assert(A.second == B.second);
        freeList(a);
        freeList(b);
    }
    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 200);
    uniform_int_distribution<int> valDist(0, 1000);
    for (int tc = 0; tc < 3000; ++tc) {
        int n = nDist(rng);
        vector<int> vals(n);
        for (int i = 0; i < n; ++i) vals[i] = valDist(rng);
        Node* a = buildList(vals);
        vector<int> rnd(n, -1);
        if (n > 0) {
            uniform_int_distribution<int> idxDist(-1, n - 1);
            for (int i = 0; i < n; ++i) rnd[i] = idxDist(rng);
        }
        assignRandom(a, rnd);
        Node* b1 = sol.copyRandomList(a);
        Node* b2 = bruteCopy(a);
        auto B1 = toVectors(b1);
        auto B2 = toVectors(b2);
        assert(B1.first == B2.first);
        assert(B1.second == B2.second);
        freeList(a);
        freeList(b1);
        freeList(b2);
    }
    cout << "All tests passed." << "\n";
    return 0;
}

