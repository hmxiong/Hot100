#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

class Trie {
public:
    Trie() {
        nodes.push_back(Node());
    }

    void insert(const string& word) {
        int cur = 0;
        for (char ch : word) {
            int c = ch - 'a';
            int nxt = nodes[cur].next[c];
            if (nxt == -1) {
                nxt = (int)nodes.size();
                nodes[cur].next[c] = nxt;
                nodes.push_back(Node());
            }
            cur = nxt;
        }
        nodes[cur].isEnd = true;
    }

    bool search(const string& word) {
        int cur = walk(word);
        return cur != -1 && nodes[cur].isEnd;
    }

    bool startsWith(const string& prefix) {
        return walk(prefix) != -1;
    }

private:
    struct Node {
        array<int, 26> next;
        bool isEnd;
        Node() : isEnd(false) {
            next.fill(-1);
        }
    };

    vector<Node> nodes;

    int walk(const string& s) {
        int cur = 0;
        for (char ch : s) {
            int c = ch - 'a';
            int nxt = nodes[cur].next[c];
            if (nxt == -1) return -1;
            cur = nxt;
        }
        return cur;
    }
};

static string randomWord(mt19937& rng, int len) {
    uniform_int_distribution<int> dist(0, 25);
    string s;
    s.reserve((size_t)len);
    for (int i = 0; i < len; ++i) s.push_back((char)('a' + dist(rng)));
    return s;
}

static bool bruteStartsWith(const unordered_set<string>& dict, const string& pref) {
    for (const auto& w : dict) {
        if (w.size() < pref.size()) continue;
        bool ok = true;
        for (size_t i = 0; i < pref.size(); ++i) {
            if (w[i] != pref[i]) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
    }
    return false;
}

int main() {
    {
        Trie trie;
        trie.insert("apple");
        assert(trie.search("apple") == true);
        assert(trie.search("app") == false);
        assert(trie.startsWith("app") == true);
        trie.insert("app");
        assert(trie.search("app") == true);
    }

    mt19937 rng(20260401);
    uniform_int_distribution<int> opDist(0, 2);   // 0 insert, 1 search, 2 startsWith
    uniform_int_distribution<int> lenDist(1, 30);

    for (int tc = 0; tc < 200; ++tc) {
        Trie trie;
        unordered_set<string> dict;

        for (int i = 0; i < 3000; ++i) {
            int op = opDist(rng);
            string s = randomWord(rng, lenDist(rng));
            if (op == 0) {
                trie.insert(s);
                dict.insert(s);
            } else if (op == 1) {
                bool got = trie.search(s);
                bool want = dict.count(s) > 0;
                assert(got == want);
            } else {
                bool got = trie.startsWith(s);
                bool want = bruteStartsWith(dict, s);
                assert(got == want);
            }
        }
    }

    cout << "All tests passed." << "\n";
    return 0;
}

