#include <cassert>
#include <iostream>
#include <vector>

using namespace std;

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x) : val(x), next(nullptr) {}
};

class Solution {
public:
    ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
        ListNode* a = headA;
        ListNode* b = headB;
        while (a != b) {
            a = a ? a->next : headB;
            b = b ? b->next : headA;
        }
        return a;
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

static void appendTail(ListNode* head, ListNode* tailHead) {
    if (!head) return;
    while (head->next) head = head->next;
    head->next = tailHead;
}

static void freeUnique(ListNode* headA, ListNode* headB, ListNode* inter) {
    vector<ListNode*> toFree;
    {
        ListNode* cur = headA;
        while (cur && cur != inter) {
            toFree.push_back(cur);
            cur = cur->next;
        }
    }
    {
        ListNode* cur = headB;
        while (cur && cur != inter) {
            toFree.push_back(cur);
            cur = cur->next;
        }
    }
    {
        ListNode* cur = inter;
        while (cur) {
            toFree.push_back(cur);
            cur = cur->next;
        }
    }
    for (auto* p : toFree) delete p;
}

int main() {
    Solution sol;

    {
        ListNode* common = buildList({8, 4, 5});
        ListNode* a = buildList({4, 1});
        ListNode* b = buildList({5, 6, 1});
        appendTail(a, common);
        appendTail(b, common);
        ListNode* got = sol.getIntersectionNode(a, b);
        assert(got == common);
        freeUnique(a, b, common);
    }
    {
        ListNode* common = buildList({2, 4});
        ListNode* a = buildList({1, 9, 1});
        ListNode* b = buildList({3});
        appendTail(a, common);
        appendTail(b, common);
        ListNode* got = sol.getIntersectionNode(a, b);
        assert(got == common);
        freeUnique(a, b, common);
    }
    {
        ListNode* a = buildList({2, 6, 4});
        ListNode* b = buildList({1, 5});
        ListNode* got = sol.getIntersectionNode(a, b);
        assert(got == nullptr);
        freeUnique(a, b, nullptr);
    }
    {
        ListNode* common = buildList({7});
        ListNode* a = common;
        ListNode* b = buildList({1, 2, 3});
        appendTail(b, common);
        ListNode* got = sol.getIntersectionNode(a, b);
        assert(got == common);
        freeUnique(a, b, common);
    }
    {
        ListNode* a = buildList({1});
        ListNode* b = a;
        ListNode* got = sol.getIntersectionNode(a, b);
        assert(got == a);
        freeUnique(a, b, a);
    }

    cout << "All tests passed." << "\n";
    return 0;
}
