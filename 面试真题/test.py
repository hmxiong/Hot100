import sys
from math import isqrt

def sieve(limit: int):
    bs = bytearray(b"\x01") * (limit + 1)
    bs[0] = bs[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if bs[i]:
            step = i
            start = i * i
            bs[start:limit + 1:step] = b"\x00" * (((limit - start) // step) + 1)
    return [i for i in range(2, limit + 1) if bs[i]]

PRIMES = sieve(31623)

def factorize(x: int, cache: dict):
    if x in cache:
        return cache[x]
    n = x
    res = {}
    for p in PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            c = 0
            while n % p == 0:
                n //= p
                c += 1
            res[p] = c
    if n > 1:
        res[n] = res.get(n, 0) + 1
    cache[x] = res
    return res

data = list(map(int, sys.stdin.buffer.read().split()))
it = iter(data)
T = next(it)
out_lines = []
cache = {}

for _ in range(T):
    n = next(it)
    arr = [next(it) for _ in range(n)]

    facts = []
    gmax = {}
    for a in arr:
        f = factorize(a, cache)
        facts.append(f)
        for p, c in f.items():
            if gmax.get(p, 0) < c:
                gmax[p] = c

    targets = [p for p, e in gmax.items() if e > 0]
    if not targets:
        out_lines.append("1")
        continue
    target_set = set(targets)

    coverage = []
    for f in facts:
        cov = [p for p, e in f.items() if gmax.get(p, 0) == e]
        coverage.append(cov)

    need = len(target_set)
    cnt = {}
    have = 0
    ans = n
    l = 0
    for r in range(n):
        for p in coverage[r]:
            c = cnt.get(p, 0) + 1
            cnt[p] = c
            if c == 1:
                have += 1
        while have == need and l <= r:
            cur = r - l + 1
            if cur < ans:
                ans = cur
            for p in coverage[l]:
                c = cnt[p] - 1
                cnt[p] = c
                if c == 0:
                    have -= 1
            l += 1
    out_lines.append(str(ans))

sys.stdout.write("\n".join(out_lines))
