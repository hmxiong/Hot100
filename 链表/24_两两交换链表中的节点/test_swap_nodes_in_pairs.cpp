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
    ListNode* swapPairs(ListNode* head) {
        ListNode dummy(0);
        dummy.next = head;
        ListNode* prev = &dummy;
        while (prev->next && prev->next->next) {
            ListNode* a = prev->next;
            ListNode* b = a->next;
            a->next = b->next;
            b->next = a;
            prev->next = b;
            prev = a;
        }
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

static vector<int> bruteSwapPairs(const vector<int>& v) {
    vector<int> res = v;
    for (int i = 0; i + 1 < (int)res.size(); i += 2) {
        swap(res[i], res[i + 1]);
    }
    return res;
}

int main() {
    Solution sol;

    {
        ListNode* head = buildList({1, 2, 3, 4});
        ListNode* gotHead = sol.swapPairs(head);
        assert(toVector(gotHead) == vector<int>({2, 1, 4, 3}));
        freeList(gotHead);
    }
    {
        ListNode* head = buildList({});
        ListNode* gotHead = sol.swapPairs(head);
        assert(gotHead == nullptr);
        freeList(gotHead);
    }
    {
        ListNode* head = buildList({1});
        ListNode* gotHead = sol.swapPairs(head);
        assert(toVector(gotHead) == vector<int>({1}));
        freeList(gotHead);
    }
    {
        ListNode* head = buildList({1, 2, 3});
        ListNode* gotHead = sol.swapPairs(head);
        assert(toVector(gotHead) == vector<int>({2, 1, 3}));
        freeList(gotHead);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 100);
    uniform_int_distribution<int> valDist(0, 100);
    for (int tc = 0; tc < 5000; ++tc) {
        int n = nDist(rng);
        vector<int> v(n);
        for (int i = 0; i < n; ++i) v[i] = valDist(rng);
        ListNode* head = buildList(v);
        ListNode* gotHead = sol.swapPairs(head);
        vector<int> got = toVector(gotHead);
        vector<int> want = bruteSwapPairs(v);
        assert(got == want);
        freeList(gotHead);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

