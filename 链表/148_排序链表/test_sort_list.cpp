#include <algorithm>
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
    ListNode* sortList(ListNode* head) {
        int n = 0;
        for (ListNode* p = head; p; p = p->next) n++;
        if (n <= 1) return head;

        ListNode dummy(0);
        dummy.next = head;

        for (int step = 1; step < n; step <<= 1) {
            ListNode* prev = &dummy;
            ListNode* cur = dummy.next;
            while (cur) {
                ListNode* left = cur;
                ListNode* right = split(left, step);
                cur = split(right, step);
                prev = mergeInto(prev, left, right);
            }
        }
        return dummy.next;
    }

private:
    static ListNode* split(ListNode* head, int size) {
        if (!head) return nullptr;
        for (int i = 1; i < size && head->next; ++i) head = head->next;
        ListNode* second = head->next;
        head->next = nullptr;
        return second;
    }

    static ListNode* mergeInto(ListNode* prev, ListNode* a, ListNode* b) {
        ListNode* tail = prev;
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
        while (tail->next) tail = tail->next;
        return tail;
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

int main() {
    Solution sol;

    {
        ListNode* head = buildList({4, 2, 1, 3});
        ListNode* gotHead = sol.sortList(head);
        assert(toVector(gotHead) == vector<int>({1, 2, 3, 4}));
        freeList(gotHead);
    }
    {
        ListNode* head = buildList({-1, 5, 3, 4, 0});
        ListNode* gotHead = sol.sortList(head);
        assert(toVector(gotHead) == vector<int>({-1, 0, 3, 4, 5}));
        freeList(gotHead);
    }
    {
        ListNode* head = buildList({});
        ListNode* gotHead = sol.sortList(head);
        assert(gotHead == nullptr);
        freeList(gotHead);
    }
    {
        ListNode* head = buildList({1});
        ListNode* gotHead = sol.sortList(head);
        assert(toVector(gotHead) == vector<int>({1}));
        freeList(gotHead);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 2000);
    uniform_int_distribution<int> valDist(-100000, 100000);

    for (int tc = 0; tc < 2000; ++tc) {
        int n = nDist(rng);
        vector<int> v(n);
        for (int i = 0; i < n; ++i) v[i] = valDist(rng);

        ListNode* head = buildList(v);
        ListNode* gotHead = sol.sortList(head);
        vector<int> got = toVector(gotHead);

        sort(v.begin(), v.end());
        assert(got == v);
        freeList(gotHead);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

