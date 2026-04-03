import sys

def solve():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    s = next(it).decode()
    adj = [[] for _ in range(n)]
    for _ in range(n - 1):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        adj[u].append(v)
        adj[v].append(u)
    par = [-1] * n
    st = [0]
    par[0] = 0
    while st:
        u = st.pop()
        for v in adj[u]:
            if par[v] == -1:
                par[v] = par[u] ^ 1
                st.append(v)
    cost1 = 0
    for i in range(n):
        expected = 'd' if par[i] == 0 else 'p'
        cost1 += 0 if s[i] == expected else 1
    cost2 = 0
    for i in range(n):
        expected = 'p' if par[i] == 0 else 'd'
        cost2 += 0 if s[i] == expected else 1
    ans = cost1 if cost1 < cost2 else cost2
    print(ans)

if __name__ == "__main__":
    solve()
