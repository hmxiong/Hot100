#include <cassert>
#include <iostream>
#include <list>
#include <random>
#include <unordered_map>
#include <vector>

using namespace std;

class LRUCache {
public:
    explicit LRUCache(int capacity) : cap(capacity) {}

    int get(int key) {
        auto it = pos.find(key);
        if (it == pos.end()) return -1;
        touch(it->second);
        return it->second->second;
    }

    void put(int key, int value) {
        auto it = pos.find(key);
        if (it != pos.end()) {
            it->second->second = value;
            touch(it->second);
            return;
        }

        if ((int)dq.size() == cap) {
            int oldKey = dq.back().first;
            pos.erase(oldKey);
            dq.pop_back();
        }
        dq.emplace_front(key, value);
        pos[key] = dq.begin();
    }

private:
    using Node = pair<int, int>;
    using It = list<Node>::iterator;

    int cap;
    list<Node> dq;
    unordered_map<int, It> pos;

    void touch(It it) {
        dq.splice(dq.begin(), dq, it);
    }
};

static vector<int> runExample() {
    LRUCache c(2);
    vector<int> out;
    c.put(1, 1);
    c.put(2, 2);
    out.push_back(c.get(1));
    c.put(3, 3);
    out.push_back(c.get(2));
    c.put(4, 4);
    out.push_back(c.get(1));
    out.push_back(c.get(3));
    out.push_back(c.get(4));
    return out;
}

static vector<int> bruteSim(int cap, const vector<pair<int, pair<int, int>>>& ops) {
    vector<pair<int, int>> order;
    unordered_map<int, int> kv;
    vector<int> out;

    auto touch = [&](int key) {
        for (int i = 0; i < (int)order.size(); ++i) {
            if (order[i].first == key) {
                auto node = order[i];
                order.erase(order.begin() + i);
                order.insert(order.begin(), node);
                break;
            }
        }
    };

    for (auto [type, kvp] : ops) {
        int key = kvp.first;
        int value = kvp.second;
        if (type == 0) {
            auto it = kv.find(key);
            if (it == kv.end()) {
                out.push_back(-1);
            } else {
                touch(key);
                out.push_back(it->second);
            }
        } else {
            if (kv.find(key) != kv.end()) {
                kv[key] = value;
                for (auto& p : order) {
                    if (p.first == key) {
                        p.second = value;
                        break;
                    }
                }
                touch(key);
            } else {
                if ((int)order.size() == cap) {
                    int evictKey = order.back().first;
                    kv.erase(evictKey);
                    order.pop_back();
                }
                kv[key] = value;
                order.insert(order.begin(), {key, value});
            }
        }
    }
    return out;
}

int main() {
    {
        vector<int> got = runExample();
        vector<int> want = {1, -1, -1, 3, 4};
        assert(got == want);
    }
    {
        LRUCache c(1);
        c.put(1, 10);
        assert(c.get(1) == 10);
        c.put(2, 20);
        assert(c.get(1) == -1);
        assert(c.get(2) == 20);
        c.put(2, 30);
        assert(c.get(2) == 30);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> capDist(1, 20);
    uniform_int_distribution<int> keyDist(0, 50);
    uniform_int_distribution<int> valDist(0, 100);
    uniform_int_distribution<int> opDist(0, 1);

    for (int tc = 0; tc < 2000; ++tc) {
        int cap = capDist(rng);
        LRUCache c(cap);

        vector<pair<int, pair<int, int>>> ops;
        ops.reserve(200);
        vector<int> got;
        for (int i = 0; i < 200; ++i) {
            int type = opDist(rng); // 0 get, 1 put
            int key = keyDist(rng);
            int val = valDist(rng);
            ops.push_back({type, {key, val}});
            if (type == 0) got.push_back(c.get(key));
            else c.put(key, val);
        }
        vector<int> want = bruteSim(cap, ops);
        assert(got == want);
    }

    cout << "All tests passed." << "\n";
    return 0;
}

