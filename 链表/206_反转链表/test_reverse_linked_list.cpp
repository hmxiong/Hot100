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
    ListNode* reverseList(ListNode* head) {
        ListNode* prev = nullptr;
        ListNode* cur = head;
        while (cur) {
            ListNode* nxt = cur->next;
            cur->next = prev;
            prev = cur;
            cur = nxt;
        }
        return prev;
    }
    ListNode* reverseListRec(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode* nh = reverseListRec(head->next);
        head->next->next = head;
        head->next = nullptr;
        return nh;
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
        ListNode* a = buildList({1,2,3,4,5});
        ListNode* r = sol.reverseList(a);
        assert(toVector(r) == vector<int>({5,4,3,2,1}));
        freeList(r);
    }
    {
        ListNode* a = buildList({});
        ListNode* r = sol.reverseList(a);
        assert(r == nullptr);
        freeList(r);
    }
    {
        ListNode* a = buildList({1});
        ListNode* r = sol.reverseList(a);
        assert(toVector(r) == vector<int>({1}));
        freeList(r);
    }
    {
        ListNode* a = buildList({1,2});
        ListNode* r = sol.reverseList(a);
        assert(toVector(r) == vector<int>({2,1}));
        freeList(r);
    }
    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 1000);
    uniform_int_distribution<int> valDist(-1000000, 1000000);
    for (int tc = 0; tc < 200; ++tc) {
        int n = nDist(rng);
        vector<int> v(n);
        for (int i = 0; i < n; ++i) v[i] = valDist(rng);
        ListNode* a = buildList(v);
        ListNode* r1 = sol.reverseList(a);
        vector<int> got = toVector(r1);
        vector<int> want = v;
        reverse(want.begin(), want.end());
        assert(got == want);
        ListNode* r2 = sol.reverseListRec(r1);
        vector<int> back = toVector(r2);
        assert(back == v);
        freeList(r2);
    }
    cout << "All tests passed." << "\n";
    return 0;
}

