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
    ListNode* detectCycle(ListNode* head) {
        ListNode* slow = head;
        ListNode* fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                ListNode* p1 = head;
                ListNode* p2 = slow;
                while (p1 != p2) {
                    p1 = p1->next;
                    p2 = p2->next;
                }
                return p1;
            }
        }
        return nullptr;
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

static ListNode* bruteDetectCycle(ListNode* head) {
    unordered_set<ListNode*> seen;
    ListNode* cur = head;
    while (cur) {
        if (seen.count(cur)) return cur;
        seen.insert(cur);
        cur = cur->next;
    }
    return nullptr;
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

int main() {
    Solution sol;

    {
        vector<int> vals{3,2,0,-4};
        ListNode* head = buildList(vals);
        connectTailTo(head, 1);
        ListNode* got = sol.detectCycle(head);
        assert(got == getNodeAt(head, 1));
        freeList(head);
    }
    {
        vector<int> vals{1,2};
        ListNode* head = buildList(vals);
        connectTailTo(head, 0);
        ListNode* got = sol.detectCycle(head);
        assert(got == getNodeAt(head, 0));
        freeList(head);
    }
    {
        vector<int> vals{1};
        ListNode* head = buildList(vals);
        ListNode* got = sol.detectCycle(head);
        assert(got == nullptr);
        freeList(head);
    }
    {
        vector<int> vals{};
        ListNode* head = buildList(vals);
        ListNode* got = sol.detectCycle(head);
        assert(got == nullptr);
        freeList(head);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 200);
    uniform_int_distribution<int> valDist(-10, 10);
    for (int tc = 0; tc < 2000; ++tc) {
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
        ListNode* got = sol.detectCycle(head);
        ListNode* want = bruteDetectCycle(head);
        assert(got == want);
        freeList(head);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

