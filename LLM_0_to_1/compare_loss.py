import argparse
import csv
import logging
import os
from typing import List, Tuple


logger = logging.getLogger("Prune")


def _read_loss_csv(csv_path: str) -> Tuple[List[int], List[float]]:
    iters: List[int] = []
    losses: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "loss" not in reader.fieldnames:
            raise ValueError(f"Invalid csv header (expect at least 'loss'): {csv_path}")
        has_iter = "iter" in reader.fieldnames
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            try:
                it = int(row["iter"]) if has_iter and row.get("iter") is not None else row_idx
                loss = float(row["loss"])
            except Exception as e:
                raise ValueError(f"Failed to parse row={row_idx} in {csv_path}: {row}") from e
            iters.append(it)
            losses.append(loss)
    return iters, losses


def save_loss_compare_plot(
    csv_paths: List[str],
    labels: List[str],
    out_dir: str,
    out_name: str,
    title: str,
):
    if len(csv_paths) != len(labels):
        raise ValueError("csv_paths and labels must have the same length")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{out_name}.png")

    curves = []
    for p, lab in zip(csv_paths, labels):
        iters, losses = _read_loss_csv(p)
        if not losses:
            raise ValueError(f"Empty loss csv: {p}")
        curves.append((iters, losses, lab, p))

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 4.5))
        for iters, losses, lab, _ in curves:
            plt.plot(iters, losses, linewidth=1.8, label=lab)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=160)
        plt.close()
        logger.info(f"[loss_vis_compare] saved png={png_path}")
    except Exception as e:
        logger.warning(f"[loss_vis_compare] matplotlib unavailable, skip png: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv64", type=str, required=True)
    parser.add_argument("--csv128", type=str, required=True)
    parser.add_argument("--csv256", type=str, required=True)
    parser.add_argument("--label64", type=str, default="block_size=64")
    parser.add_argument("--label128", type=str, default="block_size=128")
    parser.add_argument("--label256", type=str, default="block_size=256")
    parser.add_argument("--out_dir", type=str, default=os.path.join("log", "loss_vis_compare"))
    parser.add_argument("--out_name", type=str, default="loss_compare")
    parser.add_argument("--title", type=str, default="Training Loss Comparison")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    save_loss_compare_plot(
        csv_paths=[args.csv64, args.csv128, args.csv256],
        labels=[args.label64, args.label128, args.label256],
        out_dir=args.out_dir,
        out_name=args.out_name,
        title=args.title,
    )


if __name__ == "__main__":
    main()

