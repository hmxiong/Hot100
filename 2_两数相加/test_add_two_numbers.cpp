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
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode dummy(0);
        ListNode* tail = &dummy;
        int carry = 0;
        while (l1 || l2 || carry) {
            int sum = carry;
            if (l1) {
                sum += l1->val;
                l1 = l1->next;
            }
            if (l2) {
                sum += l2->val;
                l2 = l2->next;
            }
            carry = sum / 10;
            tail->next = new ListNode(sum % 10);
            tail = tail->next;
        }
        return dummy.next;
    }
};

static ListNode* buildList(const vector<int>& digitsRev) {
    ListNode dummy(0);
    ListNode* tail = &dummy;
    for (int d : digitsRev) {
        tail->next = new ListNode(d);
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

static vector<int> bruteAdd(const vector<int>& a, const vector<int>& b) {
    vector<int> res;
    int carry = 0;
    size_t i = 0;
    while (i < a.size() || i < b.size() || carry) {
        int sum = carry;
        if (i < a.size()) sum += a[i];
        if (i < b.size()) sum += b[i];
        res.push_back(sum % 10);
        carry = sum / 10;
        i++;
    }
    return res;
}

static vector<int> genDigitsRev(mt19937& rng, int n) {
    uniform_int_distribution<int> digitDist(0, 9);
    vector<int> v(n);
    for (int i = 0; i < n; ++i) v[i] = digitDist(rng);
    if (n > 1 && v.back() == 0) v.back() = 1;
    return v;
}

int main() {
    Solution sol;

    {
        ListNode* l1 = buildList({2, 4, 3});
        ListNode* l2 = buildList({5, 6, 4});
        ListNode* r = sol.addTwoNumbers(l1, l2);
        assert(toVector(r) == vector<int>({7, 0, 8}));
        freeList(l1);
        freeList(l2);
        freeList(r);
    }
    {
        ListNode* l1 = buildList({0});
        ListNode* l2 = buildList({0});
        ListNode* r = sol.addTwoNumbers(l1, l2);
        assert(toVector(r) == vector<int>({0}));
        freeList(l1);
        freeList(l2);
        freeList(r);
    }
    {
        ListNode* l1 = buildList({9, 9, 9, 9, 9, 9, 9});
        ListNode* l2 = buildList({9, 9, 9, 9});
        ListNode* r = sol.addTwoNumbers(l1, l2);
        assert(toVector(r) == vector<int>({8, 9, 9, 9, 0, 0, 0, 1}));
        freeList(l1);
        freeList(l2);
        freeList(r);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(1, 100);
    for (int tc = 0; tc < 5000; ++tc) {
        int n1 = nDist(rng);
        int n2 = nDist(rng);
        vector<int> a = genDigitsRev(rng, n1);
        vector<int> b = genDigitsRev(rng, n2);
        ListNode* l1 = buildList(a);
        ListNode* l2 = buildList(b);
        ListNode* r = sol.addTwoNumbers(l1, l2);
        vector<int> got = toVector(r);
        vector<int> want = bruteAdd(a, b);
        assert(got == want);
        freeList(l1);
        freeList(l2);
        freeList(r);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

