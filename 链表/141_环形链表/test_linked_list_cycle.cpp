#include <cassert>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    bool hasCycle(ListNode* head) {
        ListNode* slow = head;
        ListNode* fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) return true;
        }
        return false;
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

static ListNode* getNodeAt(ListNode* head, int idx) {
    while (idx-- > 0 && head) head = head->next;
    return head;
}

static void connectTailTo(ListNode* head, int pos) {
    if (!head || pos < 0) return;
    ListNode* entry = getNodeAt(head, pos);
    if (!entry) return;
    ListNode* tail = head;
    while (tail->next) tail = tail->next;
    tail->next = entry;
}

static ListNode* findCycleEntry(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) break;
    }
    if (!fast || !fast->next) return nullptr;
    ListNode* p1 = head;
    ListNode* p2 = slow;
    while (p1 != p2) {
        p1 = p1->next;
        p2 = p2->next;
    }
    return p1;
}

static void breakCycleIfAny(ListNode* head) {
    ListNode* entry = findCycleEntry(head);
    if (!entry) return;
    ListNode* cur = entry;
    while (cur->next != entry) cur = cur->next;
    cur->next = nullptr;
}

static void freeList(ListNode* head) {
    breakCycleIfAny(head);
    while (head) {
        ListNode* nxt = head->next;
        delete head;
        head = nxt;
    }
}

static bool bruteHasCycle(ListNode* head) {
    unordered_set<ListNode*> seen;
    while (head) {
        if (seen.count(head)) return true;
        seen.insert(head);
        head = head->next;
    }
    return false;
}

int main() {
    Solution sol;

    {
        ListNode* head = buildList({3, 2, 0, -4});
        connectTailTo(head, 1);
        assert(sol.hasCycle(head) == true);
        freeList(head);
    }
    {
        ListNode* head = buildList({1, 2});
        connectTailTo(head, 0);
        assert(sol.hasCycle(head) == true);
        freeList(head);
    }
    {
        ListNode* head = buildList({1});
        connectTailTo(head, -1);
        assert(sol.hasCycle(head) == false);
        freeList(head);
    }
    {
        ListNode* head = buildList({});
        assert(sol.hasCycle(head) == false);
        freeList(head);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 200);
    uniform_int_distribution<int> valDist(-10, 10);

    for (int tc = 0; tc < 3000; ++tc) {
        int n = nDist(rng);
        vector<int> vals(n);
        for (int i = 0; i < n; ++i) vals[i] = valDist(rng);

        ListNode* head = buildList(vals);
        int pos = -1;
        if (n > 0) {
            uniform_int_distribution<int> posDist(-1, n - 1);
            pos = posDist(rng);
        }
        connectTailTo(head, pos);

        bool got = sol.hasCycle(head);
        bool want = bruteHasCycle(head);
        assert(got == want);

        freeList(head);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

