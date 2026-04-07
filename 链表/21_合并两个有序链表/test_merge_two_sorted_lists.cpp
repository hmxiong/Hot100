#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode dummy(0);
        ListNode* tail = &dummy;
        ListNode* a = list1;
        ListNode* b = list2;
        while (a && b) {
            if (a->val <= b->val) {
                tail->next = a;
                a = a->next;
            } else {
                tail->next = b;
                b = b->next;
            }
            tail = tail->next;
        }
        tail->next = a ? a : b;
        return dummy.next;
    }
};

static ListNode* buildList(const vector<int>& vals) {
    ListNode dummy(0);
    ListNode* tail = &dummy;
    for (int x : vals) {
        tail->next = new ListNode(x);
        tail = tail->next;
    }
    return dummy.next;
}

static vector<int> toVector(ListNode* head) {
    vector<int> v;
    for (auto* p = head; p; p = p->next) v.push_back(p->val);
    return v;
}

static void freeList(ListNode* head) {
    while (head) {
        ListNode* nxt = head->next;
        delete head;
        head = nxt;
    }
}

static vector<int> mergeVector(const vector<int>& a, const vector<int>& b) {
    vector<int> res;
    res.reserve(a.size() + b.size());
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] <= b[j]) res.push_back(a[i++]);
        else res.push_back(b[j++]);
    }
    while (i < a.size()) res.push_back(a[i++]);
    while (j < b.size()) res.push_back(b[j++]);
    return res;
}

static vector<int> genSorted(mt19937& rng, int n) {
    uniform_int_distribution<int> startDist(-100, 100);
    uniform_int_distribution<int> stepDist(0, 5);
    int cur = startDist(rng);
    vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        cur += stepDist(rng);
        v[i] = cur;
    }
    return v;
}

int main() {
    Solution sol;

    {
        ListNode* a = buildList({1, 2, 4});
        ListNode* b = buildList({1, 3, 4});
        ListNode* c = sol.mergeTwoLists(a, b);
        assert(toVector(c) == vector<int>({1, 1, 2, 3, 4, 4}));
        freeList(c);
    }
    {
        ListNode* a = buildList({});
        ListNode* b = buildList({});
        ListNode* c = sol.mergeTwoLists(a, b);
        assert(c == nullptr);
        freeList(c);
    }
    {
        ListNode* a = buildList({});
        ListNode* b = buildList({0});
        ListNode* c = sol.mergeTwoLists(a, b);
        assert(toVector(c) == vector<int>({0}));
        freeList(c);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 50);
    for (int tc = 0; tc < 5000; ++tc) {
        int n1 = nDist(rng);
        int n2 = nDist(rng);
        vector<int> v1 = genSorted(rng, n1);
        vector<int> v2 = genSorted(rng, n2);

        ListNode* a = buildList(v1);
        ListNode* b = buildList(v2);
        ListNode* c = sol.mergeTwoLists(a, b);

        vector<int> got = toVector(c);
        vector<int> want = mergeVector(v1, v2);
        assert(got == want);

        freeList(c);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

