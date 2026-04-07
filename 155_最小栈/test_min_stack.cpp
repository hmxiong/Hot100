#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

class MinStack {
public:
    MinStack() = default;

    void push(int val) {
        st.push_back(val);
        if (mins.empty()) mins.push_back(val);
        else mins.push_back(std::min(mins.back(), val));
    }

    void pop() {
        st.pop_back();
        mins.pop_back();
    }

    int top() {
        return st.back();
    }

    int getMin() {
        return mins.back();
    }

private:
    vector<int> st;
    vector<int> mins;
};

static int bruteMin(const vector<int>& st) {
    return *min_element(st.begin(), st.end());
}

int main() {
    {
        MinStack s;
        s.push(-2);
        s.push(0);
        s.push(-3);
        assert(s.getMin() == -3);
        s.pop();
        assert(s.top() == 0);
        assert(s.getMin() == -2);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> valDist(-1000, 1000);
    uniform_int_distribution<int> opDist(0, 99);

    for (int tc = 0; tc < 2000; ++tc) {
        MinStack s;
        vector<int> st;
        for (int i = 0; i < 500; ++i) {
            int op = opDist(rng);
            if (st.empty() || op < 50) {
                int v = valDist(rng);
                s.push(v);
                st.push_back(v);
                assert(s.top() == st.back());
                assert(s.getMin() == bruteMin(st));
            } else if (op < 75) {
                s.pop();
                st.pop_back();
                if (!st.empty()) {
                    assert(s.top() == st.back());
                    assert(s.getMin() == bruteMin(st));
                }
            } else if (op < 90) {
                assert(s.top() == st.back());
            } else {
                assert(s.getMin() == bruteMin(st));
            }
        }
    }

    cout << "All tests passed." << "\n";
    return 0;
}

