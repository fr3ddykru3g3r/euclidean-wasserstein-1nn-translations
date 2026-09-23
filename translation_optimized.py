import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from threadpoolctl import threadpool_limits


def square(radius):
    return [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
    ]


def save_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize(images):
    masses = images.sum(
        axis=(1, 2),
        keepdims=True,
    )
    if np.any(masses <= 0):
        raise ValueError(
            "Blank images cannot be mass normalized."
        )

    return images.astype(np.float64) / masses


def choose_split(labels, args):
    if args.split:
        with np.load(
            args.split,
            allow_pickle=False,
        ) as z:
            train = z["train_indices"]
            test = z["test_indices"]
    else:
        rng = np.random.default_rng(args.seed)
        train, test = [], []

        for label in np.unique(labels):
            pool = rng.permutation(
                np.flatnonzero(labels == label)
            )
            n = args.train_per_class
            m = args.test_per_class

            if n + m > len(pool):
                raise ValueError(
                    f"Not enough examples of class {label}."
                )

            train.extend(pool[:n])
            test.extend(pool[n:n + m])

        train = np.array(train)
        test = np.array(test)

    for name, indices in [
        ("train", train),
        ("test", test),
    ]:
        if (
            indices.ndim != 1
            or not np.issubdtype(
                indices.dtype,
                np.integer,
            )
        ):
            raise ValueError(
                f"{name} indices must be a "
                "one-dimensional integer array."
            )

        if (
            len(indices) == 0
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError(
                f"{name} indices must be nonempty "
                "and unique."
            )

        if (
            np.any(indices < 0)
            or np.any(indices >= len(labels))
        ):
            raise ValueError(
                f"{name} index outside sklearn "
                "digits dataset."
            )

    if np.intersect1d(train, test).size:
        raise ValueError(
            "Train/test overlap is forbidden."
        )

    if (
        not set(labels[test]).issubset(
            set(labels[train])
        )
        or len(set(labels[train])) < 2
    ):
        raise ValueError(
            "Training set needs every test class "
            "and at least two classes."
        )

    return train, test


class TranslationDistances:
    def __init__(
        self,
        query,
        train,
        max_relative_shift,
    ):
        self.query, self.train = query, train
        self.pad = max_relative_shift

        h, w = train.shape[1:]
        self.shape = (
            h + 2 * self.pad,
            w + 2 * self.pad,
        )

        self.x = self.place(
            query,
            (0, 0),
        ).reshape(len(query), -1)

        self.xnorm = np.sum(
            self.x * self.x,
            axis=1,
        )[:, None]

        self.qnorm = np.sum(
            train * train,
            axis=(1, 2),
        )[None, :]

        self.cache = {}

    def place(self, batch, shift):
        dx, dy = shift

        if max(abs(dx), abs(dy)) > self.pad:
            raise ValueError(
                "Insufficient canvas padding."
            )

        h, w = batch.shape[1:]
        canvas = np.zeros(
            (len(batch), *self.shape)
        )

        y = self.pad + dy
        x = self.pad + dx
        canvas[:, y:y + h, x:x + w] = batch

        return canvas

    def pairwise(self, relative_shift):
        if relative_shift not in self.cache:
            q = self.place(
                self.train,
                relative_shift,
            ).reshape(len(self.train), -1)

            squared = (
                self.xnorm
                + self.qnorm
                - 2 * (self.x @ q.T)
            )

            self.cache[relative_shift] = np.sqrt(
                np.maximum(squared, 0)
            )

        return self.cache[relative_shift]

    def scores(self, radius, query_shift=(0, 0)):
        ux, uy = query_shift
        best = np.full(
            (len(self.query), len(self.train)),
            np.inf,
        )

        for tx, ty in square(radius):
            np.minimum(
                best,
                self.pairwise((tx - ux, ty - uy)),
                out=best,
            )

        return best


def class_margin(
    scores,
    train_labels,
    target_labels,
):
    same = (
        target_labels[:, None]
        == train_labels[None, :]
    )

    a = np.min(
        np.where(same, scores, np.inf),
        axis=1,
    )
    b = np.min(
        np.where(~same, scores, np.inf),
        axis=1,
    )

    return a, b, b - a


def transport_scores(query, train, conditions):
    try:
        import ot
    except ImportError as exc:
        raise SystemExit(
            "Wasserstein requires POT: "
            "python -m pip install POT\n"
            "Or run with --skip-wasserstein."
        ) from exc

    def support(image):
        locations = np.argwhere(image > 0)
        weights = np.ascontiguousarray(
            image[image > 0],
            dtype=np.float64,
        )
        return (
            locations[:, ::-1].astype(float),
            weights,
        )

    train_supports = [
        support(q) for q in train
    ]
    query_supports = [
        support(x) for x in query
    ]

    result = {}
    total = (
        len(query) * len(train) * len(conditions)
    )
    print(
        f"Exact Wasserstein: {total:,} "
        "transport problems; this is the slow step.",
        flush=True,
    )

    for name, shift in conditions:
        scores = np.empty(
            (len(query), len(train))
        )

        for i, (xp, a) in enumerate(query_supports):
            for j, (qp, b) in enumerate(train_supports):
                costs = cdist(
                    xp + np.array(shift),
                    qp,
                    metric="euclidean",
                )
                scores[i, j] = ot.emd2(
                    a,
                    b,
                    costs,
                    numItermax=1000000,
                    numThreads=1,
                )

            if (
                (i + 1) % 10 == 0
                or i + 1 == len(query)
            ):
                print(
                    f"  {name}: "
                    f"{i + 1}/{len(query)} queries",
                    flush=True,
                )

        result[name] = scores

    return result


def main():
    p = argparse.ArgumentParser(
        description=(
            "Compare Euclidean, Wasserstein and "
            "translation-optimized 1-NN"
        ),
        formatter_class=(
            argparse.RawDescriptionHelpFormatter
        ),
    )

    p.add_argument(
        "--train-per-class",
        type=int,
        default=30,
    )
    p.add_argument(
        "--test-per-class",
        type=int,
        default=15,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--radii",
        type=int,
        nargs="+",
        default=[1, 2],
    )
    p.add_argument(
        "--certificate-shift",
        type=int,
        default=2,
    )
    p.add_argument(
        "--split",
        type=Path,
        help=(
            "NPZ with train_indices and test_indices "
            "for sklearn digits"
        ),
    )
    p.add_argument(
        "--skip-wasserstein",
        action="store_true",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("toe_results"),
    )

    args = p.parse_args()

    if (
        min(
            args.train_per_class,
            args.test_per_class,
        ) < 1
        or min(args.radii) < 0
    ):
        p.error(
            "Sample counts must be positive "
            "and radii nonnegative."
        )

    if args.certificate_shift < 1:
        p.error(
            "--certificate-shift must be at least 1."
        )

    args.radii = sorted(set(args.radii))

    if (
        args.out.exists()
        and any(args.out.iterdir())
    ):
        p.error(
            "Output directory is nonempty; "
            "choose a new --out directory."
        )

    if not args.skip_wasserstein:
        try:
            import ot
        except ImportError:
            p.error(
                "Install POT with: "
                "python -m pip install POT; "
                "or use --skip-wasserstein"
            )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    start = time.perf_counter()
    data = load_digits()

    train_indices, test_indices = choose_split(
        data.target,
        args,
    )

    train = normalize(data.images[train_indices])
    query = normalize(data.images[test_indices])
    ytrain = data.target[train_indices]
    ytest = data.target[test_indices]

    np.savez(
        args.out / "split_indices.npz",
        train_indices=train_indices,
        test_indices=test_indices,
    )

    print(
        f"Train={len(train)}, test={len(query)}. "
        "Same split for every method.",
        flush=True,
    )

    if not args.split:
        print(
            "Stratified split generated with "
            f"seed {args.seed}.",
            flush=True,
        )

    conditions = [("clean", (0, 0))]

    for k in [1, 2]:
        conditions.extend([
            (f"right_{k}", (k, 0)),
            (f"left_{k}", (-k, 0)),
            (f"down_{k}", (0, k)),
            (f"up_{k}", (0, -k)),
        ])

    radii = {
        "Euclidean": 0,
        **{f"TOE_r{r}": r for r in args.radii},
    }

    pad = (
        max(args.radii + [0])
        + max(2, args.certificate_shift)
    )

    bank = TranslationDistances(
        query,
        train,
        pad,
    )

    predictions = {}
    scores_by_method = {}

    with threadpool_limits(limits=1):
        direct = cdist(
            query.reshape(len(query), -1),
            train.reshape(len(train), -1),
        )
        np.testing.assert_allclose(
            bank.scores(0),
            direct,
            atol=1e-7,
            rtol=1e-7,
        )

        for method, r in radii.items():
            print(
                f"Computing {method}...",
                flush=True,
            )
            scores_by_method[method] = {
                name: bank.scores(r, shift)
                for name, shift in conditions
            }

        if not args.skip_wasserstein:
            scores_by_method["Wasserstein_W1"] = (
                transport_scores(
                    query,
                    train,
                    conditions,
                )
            )

        details, summary = [], []

        for method, scores in scores_by_method.items():
            nearest = np.stack([
                scores[name].argmin(axis=1)
                for name, _ in conditions
            ])

            pred = ytrain[nearest]
            predictions[method] = pred
            correct = pred == ytest[None, :]

            clean = 100 * correct[0].mean()
            one = 100 * correct[1:5].mean()
            two = 100 * correct[5:9].mean()

            row = dict(
                method=method,
                clean_accuracy_pct=clean,
                one_pixel_mean_accuracy_pct=one,
                two_pixel_mean_accuracy_pct=two,
                one_pixel_loss_pp=clean - one,
                two_pixel_loss_pp=clean - two,
                worst_direction_accuracy_pct=(
                    100 * correct.mean(axis=1).min()
                ),
                all_nine_correct_pct=(
                    100 * correct.all(axis=0).mean()
                ),
                all_nine_label_stable_pct=(
                    100
                    * (pred == pred[0]).all(axis=0).mean()
                ),
            )
            summary.append(row)

            for k, (name, shift) in enumerate(conditions):
                _, _, margin = class_margin(
                    scores[name],
                    ytrain,
                    ytest,
                )

                for i in range(len(query)):
                    details.append(
                        dict(
                            method=method,
                            condition=name,
                            dx=shift[0],
                            dy=shift[1],
                            test_index=int(test_indices[i]),
                            true_label=int(ytest[i]),
                            predicted_label=int(pred[k, i]),
                            correct=bool(correct[k, i]),
                            nearest_train_index=int(
                                train_indices[nearest[k, i]]
                            ),
                            true_class_margin=float(
                                margin[i]
                            ),
                        )
                    )

        save_csv(
            args.out / "accuracy_summary.csv",
            summary,
        )
        save_csv(
            args.out / "predictions.csv",
            details,
        )

        certificates, cert_summary = [], []
        s = args.certificate_shift
        tol = 1e-7

        for r in args.radii:
            if r < s:
                print(
                    f"TOE_r{r}: theorem certificate "
                    f"unavailable because r < s={s}.",
                    flush=True,
                )
                continue

            print(
                f"Checking TOE_r{r} on all "
                f"{(2 * s + 1) ** 2} square shifts...",
                flush=True,
            )

            a, _, _ = class_margin(
                bank.scores(r - s),
                ytrain,
                ytest,
            )
            _, b, _ = class_margin(
                bank.scores(r + s),
                ytrain,
                ytest,
            )

            gap = b - a
            worst_margin = np.full(
                len(query),
                np.inf,
            )
            all_correct = np.ones(
                len(query),
                dtype=bool,
            )

            for u in square(s):
                scores = bank.scores(r, u)

                _, _, margin = class_margin(
                    scores,
                    ytrain,
                    ytest,
                )

                worst_margin = np.minimum(
                    worst_margin,
                    margin,
                )
                all_correct &= (
                    ytrain[scores.argmin(axis=1)]
                    == ytest
                )

            certified = gap > tol
            violations = certified & ~all_correct

            if (
                violations.any()
                or np.any(worst_margin < gap - tol)
            ):
                raise AssertionError(
                    "Certificate violated: inspect "
                    "translation implementation."
                )

            cert_summary.append(
                dict(
                    radius=r,
                    shift_bound=s,
                    certified_correct_pct=(
                        100 * certified.mean()
                    ),
                    actual_all_square_shifts_correct_pct=(
                        100 * all_correct.mean()
                    ),
                    strict_all_square_shifts_correct_pct=(
                        100 * (worst_margin > tol).mean()
                    ),
                    certificate_violations=int(
                        violations.sum()
                    ),
                )
            )

            for i in range(len(query)):
                certificates.append(
                    dict(
                        test_index=int(test_indices[i]),
                        radius=r,
                        shift_bound=s,
                        A_inner=float(a[i]),
                        B_outer=float(b[i]),
                        certificate_gap=float(gap[i]),
                        certified_correct=bool(
                            certified[i]
                        ),
                        actual_worst_margin=float(
                            worst_margin[i]
                        ),
                        all_square_shifts_correct=bool(
                            all_correct[i]
                        ),
                    )
                )

        save_csv(
            args.out / "certificates.csv",
            certificates,
        )
        save_csv(
            args.out / "certificate_summary.csv",
            cert_summary,
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for row in summary:
        ax.plot(
            [0, 1, 2],
            [
                row["clean_accuracy_pct"],
                row["one_pixel_mean_accuracy_pct"],
                row["two_pixel_mean_accuracy_pct"],
            ],
            marker="o",
            label=row["method"],
        )

    ax.set(
        xlabel=(
            "Query translation "
            "(pixels; cardinal directions averaged)"
        ),
        ylabel="Accuracy (%)",
        xticks=[0, 1, 2],
        ylim=(0, 100),
        title=(
            "Euclidean, Wasserstein "
            "and translation search"
        ),
    )
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        args.out / "accuracy_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    versions = {}
    for package in [
        "numpy",
        "scipy",
        "scikit-learn",
        "POT",
        "matplotlib",
    ]:
        try:
            versions[package] = (
                importlib.metadata.version(package)
            )
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    metadata = dict(
        dataset="sklearn.datasets.load_digits",
        dataset_sha256=hashlib.sha256(
            data.images.tobytes()
            + data.target.tobytes()
        ).hexdigest(),
        seed=args.seed if not args.split else None,
        split_source=(
            str(args.split)
            if args.split
            else "Seeded stratified sample"
        ),
        n_train=len(train),
        n_test=len(query),
        radii=args.radii,
        certificate_shift=s,
        normalization=(
            "Each image divided by its total brightness"
        ),
        shifts=conditions,
        translation_model=(
            "Whole lattice, zero extension, no clipping"
        ),
        transport=(
            "Exact balanced W1, "
            "Euclidean pixel-coordinate ground cost"
        ),
        tie_breaking=(
            "First training index in saved split order "
            "(numpy argmin)"
        ),
        certificate_tolerance=tol,
        padding_per_side=pad,
        canvas_shape=bank.shape,
        versions=versions,
        elapsed_seconds=time.perf_counter() - start,
        timing_note=(
            "Shared cached distances: "
            "not a per-method speed benchmark"
        ),
        scope=(
            "Finite test-set results; "
            "no population or novelty guarantee"
        ),
    )

    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print(
        "\nMethod                 Clean     "
        "1 pixel    2 pixels   All 9 correct"
    )

    for row in summary:
        print(
            f"{row['method']:<22} "
            f"{row['clean_accuracy_pct']:7.2f}%"
            f"  {row['one_pixel_mean_accuracy_pct']:7.2f}%"
            f"  {row['two_pixel_mean_accuracy_pct']:7.2f}%"
            f"  {row['all_nine_correct_pct']:7.2f}%"
        )

    for row in cert_summary:
        print(
            f"r={row['radius']}, s={s}: "
            f"certified {row['certified_correct_pct']:.2f}%; "
            "actually correct on every square shift "
            f"{row['actual_all_square_shifts_correct_pct']:.2f}%; "
            f"violations={row['certificate_violations']}"
        )

    print(f"\nResults saved to {args.out.resolve()}")


if __name__ == "__main__":
    main()
