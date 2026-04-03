import sys

def solve():
    data = list(map(int, sys.stdin.buffer.read().split()))
    if len(data) != 4:
        return
    n, m, p, q = data
    yearly = m * (12 + q)
    full_years = n // yearly
    remaining = n - full_years * yearly
    if remaining == 0:
        print(full_years * 12)
        return
    start = p
    end = p + q - 1
    acc = 0
    months = 0
    for mon in range(1, 13):
        drop = 2 * m if (start <= mon <= end) else m
        acc += drop
        months += 1
        if acc >= remaining:
            break
    print(full_years * 12 + months)

if __name__ == "__main__":
    solve()
