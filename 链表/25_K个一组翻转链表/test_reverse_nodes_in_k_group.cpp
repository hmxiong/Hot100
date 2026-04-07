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
    ListNode* reverseKGroup(ListNode* head, int k) {
        if (k <= 1 || !head) return head;

        ListNode dummy(0);
        dummy.next = head;
        ListNode* groupPrev = &dummy;

        while (true) {
            ListNode* kth = groupPrev;
            for (int i = 0; i < k && kth; ++i) kth = kth->next;
            if (!kth) break;

            ListNode* groupNext = kth->next;

            ListNode* prev = groupNext;
            ListNode* cur = groupPrev->next;
            while (cur != groupNext) {
                ListNode* nxt = cur->next;
                cur->next = prev;
                prev = cur;
                cur = nxt;
            }

            ListNode* newGroupHead = prev;
            ListNode* newGroupTail = groupPrev->next;
            groupPrev->next = newGroupHead;
            groupPrev = newGroupTail;
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

static vector<int> bruteReverseKGroup(const vector<int>& v, int k) {
    vector<int> res = v;
    int n = (int)res.size();
    for (int i = 0; i + k <= n; i += k) {
        int l = i;
        int r = i + k - 1;
        while (l < r) {
            swap(res[l], res[r]);
            l++;
            r--;
        }
    }
    return res;
}

int main() {
    Solution sol;

    {
        ListNode* head = buildList({1, 2, 3, 4, 5});
        ListNode* got = sol.reverseKGroup(head, 2);
        assert(toVector(got) == vector<int>({2, 1, 4, 3, 5}));
        freeList(got);
    }
    {
        ListNode* head = buildList({1, 2, 3, 4, 5});
        ListNode* got = sol.reverseKGroup(head, 3);
        assert(toVector(got) == vector<int>({3, 2, 1, 4, 5}));
        freeList(got);
    }
    {
        ListNode* head = buildList({1});
        ListNode* got = sol.reverseKGroup(head, 1);
        assert(toVector(got) == vector<int>({1}));
        freeList(got);
    }
    {
        ListNode* head = buildList({1, 2, 3});
        ListNode* got = sol.reverseKGroup(head, 5);
        assert(toVector(got) == vector<int>({1, 2, 3}));
        freeList(got);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(1, 200);
    uniform_int_distribution<int> valDist(0, 1000);

    for (int tc = 0; tc < 3000; ++tc) {
        int n = nDist(rng);
        vector<int> v(n);
        for (int i = 0; i < n; ++i) v[i] = valDist(rng);
        uniform_int_distribution<int> kDist(1, n + 3);
        int k = kDist(rng);

        ListNode* head = buildList(v);
        ListNode* gotHead = sol.reverseKGroup(head, k);
        vector<int> got = toVector(gotHead);
        vector<int> want = bruteReverseKGroup(v, k);
        assert(got == want);
        freeList(gotHead);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

