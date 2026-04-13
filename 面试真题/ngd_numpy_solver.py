import json
import sys

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        return
    data = json.loads(raw)

    if np is None:
        import math

        train_x = [[float(v) for v in row] for row in data["train_X"]]
        train_y = [float(v) for v in data["train_y"]]
        test_x = [[float(v) for v in row] for row in data["test_X"]]

        d = len(train_x[0]) if train_x else 0
        w = [0.0] * d

        eta0 = 0.2
        lam = 0.01
        eps = 1e-10
        t_max = 60

        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        def matvec(a, x):
            return [dot(row, x) for row in a]

        xt = list(zip(*train_x)) if train_x else []

        for t in range(1, t_max + 1):
            r = [ax - y for ax, y in zip(matvec(train_x, w), train_y)]
            grad = [dot(col, r) + 2.0 * lam * wi for col, wi in zip(xt, w)]
            if not all(math.isfinite(g) for g in grad):
                continue
            norm = math.sqrt(sum(g * g for g in grad))
            if (not math.isfinite(norm)) or norm < eps:
                continue
            lr = eta0 / math.sqrt(t)
            scale = lr / max(norm, eps)
            w = [wi - scale * gi for wi, gi in zip(w, grad)]

        out = []
        for row in test_x:
            out.append(1 if dot(row, w) >= 0.0 else 0)
        result = {"weights": [round(wi, 6) for wi in w], "test_pred": out}
        sys.stdout.write(json.dumps(result, ensure_ascii=False))
        return

    train_x = np.asarray(data["train_X"], dtype=np.float64)
    train_y = np.asarray(data["train_y"], dtype=np.float64)
    test_x = np.asarray(data["test_X"], dtype=np.float64)

    if train_x.ndim != 2:
        train_x = train_x.reshape((train_x.shape[0], -1))
    if test_x.ndim != 2:
        test_x = test_x.reshape((test_x.shape[0], -1))

    d = train_x.shape[1]
    w = np.zeros(d, dtype=np.float64)

    eta0 = 0.2
    lam = 0.01
    eps = 1e-10
    t_max = 60

    xt = train_x.T
    for t in range(1, t_max + 1):
        r = train_x @ w - train_y
        grad = xt @ r + 2.0 * lam * w
        if not np.all(np.isfinite(grad)):
            continue
        norm = float(np.linalg.norm(grad))
        if (not np.isfinite(norm)) or norm < eps:
            continue
        lr = eta0 / np.sqrt(t)
        w = w - lr * (grad / max(norm, eps))

    logits = test_x @ w
    pred = (logits >= 0.0).astype(np.int64)
    result = {"weights": [round(float(wi), 6) for wi in w.tolist()], "test_pred": pred.tolist()}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
