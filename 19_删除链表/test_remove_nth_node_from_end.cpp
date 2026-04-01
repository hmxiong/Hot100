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
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy(0);
        dummy.next = head;

        ListNode* fast = &dummy;
        ListNode* slow = &dummy;
        for (int i = 0; i < n; ++i) fast = fast->next;
        while (fast->next) {
            fast = fast->next;
            slow = slow->next;
        }
        ListNode* del = slow->next;
        slow->next = del->next;
        delete del;
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

static vector<int> bruteRemoveNthFromEnd(const vector<int>& v, int n) {
    vector<int> res = v;
    int idx = (int)res.size() - n;
    res.erase(res.begin() + idx);
    return res;
}

int main() {
    Solution sol;

    {
        ListNode* head = buildList({1, 2, 3, 4, 5});
        ListNode* gotHead = sol.removeNthFromEnd(head, 2);
        assert(toVector(gotHead) == vector<int>({1, 2, 3, 5}));
        freeList(gotHead);
    }
    {
        ListNode* head = buildList({1});
        ListNode* gotHead = sol.removeNthFromEnd(head, 1);
        assert(gotHead == nullptr);
        freeList(gotHead);
    }
    {
        ListNode* head = buildList({1, 2});
        ListNode* gotHead = sol.removeNthFromEnd(head, 1);
        assert(toVector(gotHead) == vector<int>({1}));
        freeList(gotHead);
    }
    {
        ListNode* head = buildList({1, 2});
        ListNode* gotHead = sol.removeNthFromEnd(head, 2);
        assert(toVector(gotHead) == vector<int>({2}));
        freeList(gotHead);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(1, 30);
    uniform_int_distribution<int> valDist(0, 100);

    for (int tc = 0; tc < 5000; ++tc) {
        int sz = nDist(rng);
        vector<int> v(sz);
        for (int i = 0; i < sz; ++i) v[i] = valDist(rng);
        uniform_int_distribution<int> kDist(1, sz);
        int n = kDist(rng);

        ListNode* head = buildList(v);
        ListNode* gotHead = sol.removeNthFromEnd(head, n);
        vector<int> got = toVector(gotHead);
        vector<int> want = bruteRemoveNthFromEnd(v, n);
        assert(got == want);
        freeList(gotHead);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

