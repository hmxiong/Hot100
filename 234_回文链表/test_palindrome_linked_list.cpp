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
    bool isPalindrome(ListNode* head) {
        if (!head || !head->next) return true;

        ListNode* slow = head;
        ListNode* fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        if (fast) slow = slow->next;

        ListNode* second = reverseList(slow);
        ListNode* secondHead = second;s
        ListNode* first = head;
        bool ok = true;
        while (second) {
            if (first->val != second->val) {
                ok = false;
                break;
            }
            first = first->next;
            second = second->next;
        }
        reverseList(secondHead);
        return ok;
    }

private:
    static ListNode* reverseList(ListNode* head) {
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

static bool bruteIsPalindrome(const vector<int>& v) {
    int i = 0;
    int j = (int)v.size() - 1;
    while (i < j) {
        if (v[i] != v[j]) return false;
        i++;
        j--;
    }
    return true;
}

int main() {
    Solution sol;

    {
        ListNode* a = buildList({1, 2, 2, 1});
        assert(sol.isPalindrome(a) == true);
        assert(toVector(a) == vector<int>({1, 2, 2, 1}));
        freeList(a);
    }
    {
        ListNode* a = buildList({1, 2});
        assert(sol.isPalindrome(a) == false);
        assert(toVector(a) == vector<int>({1, 2}));
        freeList(a);
    }
    {
        ListNode* a = buildList({});
        assert(sol.isPalindrome(a) == true);
        freeList(a);
    }
    {
        ListNode* a = buildList({7});
        assert(sol.isPalindrome(a) == true);
        assert(toVector(a) == vector<int>({7}));
        freeList(a);
    }
    {
        ListNode* a = buildList({1, 2, 3, 2, 1});
        assert(sol.isPalindrome(a) == true);
        assert(toVector(a) == vector<int>({1, 2, 3, 2, 1}));
        freeList(a);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> nDist(0, 200);
    uniform_int_distribution<int> valDist(-10, 10);
    for (int tc = 0; tc < 2000; ++tc) {
        int n = nDist(rng);
        vector<int> v(n);
        for (int i = 0; i < n; ++i) v[i] = valDist(rng);
        ListNode* a = buildList(v);
        bool got = sol.isPalindrome(a);
        bool want = bruteIsPalindrome(v);
        assert(got == want);
        assert(toVector(a) == v);
        freeList(a);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

