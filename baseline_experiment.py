from __future__ import annotations

import sys
import json
import platform
import importlib.metadata
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_digits
    from sklearn.metrics import confusion_matrix
except ImportError as exc:
    missing = getattr(exc, "name", "a required package")
    raise SystemExit(
        f"Missing package: {missing}\n"
        "Install the dependencies with:\n"
        "python3 -m pip install numpy pandas matplotlib "
        "scikit-learn opencv-python"
    ) from exc


QUICK_MODE = False
RANDOM_SEED = 42
CANVAS_SIZE = 12
BOOTSTRAP_REPETITIONS = 2000
OUTPUT_DIR = Path("wasserstein_1nn_results")

if QUICK_MODE:
    TRAIN_PER_CLASS = 5
    TEST_PER_CLASS = 3
    TRANSLATIONS = [
        ("clean", 0, 0, 0),
        ("right_1", 1, 0, 1),
        ("down_1", 1, 1, 0),
        ("right_2", 2, 0, 2),
        ("down_2", 2, 2, 0),
    ]
else:
    TRAIN_PER_CLASS = 30
    TEST_PER_CLASS = 15
    TRANSLATIONS = [
        ("clean", 0, 0, 0),
        ("left_1", 1, 0, -1),
        ("right_1", 1, 0, 1),
        ("up_1", 1, -1, 0),
        ("down_1", 1, 1, 0),
        ("left_2", 2, 0, -2),
        ("right_2", 2, 0, 2),
        ("up_2", 2, -2, 0),
        ("down_2", 2, 2, 0),
    ]


@dataclass(frozen=True)
class ImageRepresentation:
    image: np.ndarray
    flat: np.ndarray
    signature: np.ndarray
    centroid: np.ndarray


def choose_balanced_samples(
    images: np.ndarray,
    labels: np.ndarray,
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []

    for digit in range(10):
        class_indices = np.flatnonzero(labels == digit)
        rng.shuffle(class_indices)

        required = train_per_class + test_per_class
        if len(class_indices) < required:
            raise ValueError(
                f"Digit {digit} has only "
                f"{len(class_indices)} samples; "
                f"{required} were requested."
            )

        train_indices.extend(
            class_indices[:train_per_class]
        )
        test_indices.extend(
            class_indices[train_per_class:required]
        )

    train_indices = np.asarray(train_indices)
    test_indices = np.asarray(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    np.savez(
        OUTPUT_DIR / "split_indices.npz",
        train_indices=train_indices,
        test_indices=test_indices,
    )

    return (
        images[train_indices],
        labels[train_indices],
        images[test_indices],
        labels[test_indices],
    )


def pad_image(
    image: np.ndarray,
    canvas_size: int,
) -> np.ndarray:
    height, width = image.shape
    if canvas_size < height or canvas_size < width:
        raise ValueError(
            "The canvas must be at least as large as the image."
        )

    padded = np.zeros(
        (canvas_size, canvas_size),
        dtype=np.float64,
    )
    row_start = (canvas_size - height) // 2
    col_start = (canvas_size - width) // 2

    padded[
        row_start:row_start + height,
        col_start:col_start + width,
    ] = image

    return padded


def mass_normalize(image: np.ndarray) -> np.ndarray:
    if np.any(image < 0):
        raise ValueError(
            "Pixel intensities must be non-negative."
        )

    total = float(image.sum())
    if total <= 0:
        raise ValueError(
            "Cannot mass-normalize an image "
            "with zero total intensity."
        )

    return image.astype(np.float64, copy=False) / total


def translate_image(
    image: np.ndarray,
    dy: int,
    dx: int,
) -> np.ndarray:
    height, width = image.shape
    translated = np.zeros_like(image)

    source_y_start = max(0, -dy)
    source_y_end = min(height, height - dy)
    source_x_start = max(0, -dx)
    source_x_end = min(width, width - dx)

    destination_y_start = max(0, dy)
    destination_y_end = min(height, height + dy)
    destination_x_start = max(0, dx)
    destination_x_end = min(width, width + dx)

    if (
        source_y_start < source_y_end
        and source_x_start < source_x_end
    ):
        translated[
            destination_y_start:destination_y_end,
            destination_x_start:destination_x_end,
        ] = image[
            source_y_start:source_y_end,
            source_x_start:source_x_end,
        ]

    if not np.isclose(
        translated.sum(),
        image.sum(),
        atol=1e-12,
    ):
        raise ValueError(
            f"Translation (dy={dy}, dx={dx}) "
            "cropped image mass. Increase CANVAS_SIZE."
        )

    return translated


def image_to_signature(
    image: np.ndarray,
    tolerance: float = 1e-15,
) -> np.ndarray:
    rows, cols = np.nonzero(image > tolerance)
    weights = image[rows, cols]

    signature = np.column_stack(
        (weights, rows, cols)
    ).astype(np.float32)

    if signature.size == 0:
        raise ValueError(
            "The EMD signature cannot be empty."
        )

    signature[:, 0] /= signature[:, 0].sum()
    return signature


def signature_centroid(
    signature: np.ndarray,
) -> np.ndarray:
    weights = signature[:, 0].astype(np.float64)
    coordinates = signature[:, 1:].astype(np.float64)
    return np.sum(
        weights[:, None] * coordinates,
        axis=0,
    )


def represent_image(
    image: np.ndarray,
) -> ImageRepresentation:
    normalized = mass_normalize(image)
    signature = image_to_signature(normalized)

    return ImageRepresentation(
        image=normalized,
        flat=normalized.ravel(),
        signature=signature,
        centroid=signature_centroid(signature),
    )


def exact_wasserstein_distance(
    first_signature: np.ndarray,
    second_signature: np.ndarray,
) -> float:
    distance, _, _ = cv2.EMD(
        first_signature,
        second_signature,
        cv2.DIST_L2,
    )
    return float(distance)


def euclidean_1nn(
    query_flat: np.ndarray,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[int, int, float]:
    distances = np.linalg.norm(
        train_matrix - query_flat,
        axis=1,
    )
    neighbour_index = int(np.argmin(distances))

    return (
        int(train_labels[neighbour_index]),
        neighbour_index,
        float(distances[neighbour_index]),
    )


def wasserstein_1nn(
    query: ImageRepresentation,
    train_representations: list[ImageRepresentation],
    train_labels: np.ndarray,
) -> tuple[int, int, float, int]:
    train_centroids = np.vstack(
        [item.centroid for item in train_representations]
    )
    lower_bounds = np.linalg.norm(
        train_centroids - query.centroid,
        axis=1,
    )
    candidate_order = np.argsort(lower_bounds)

    best_index = -1
    best_distance = np.inf
    emd_evaluations = 0

    for candidate_index in candidate_order:
        lower_bound = float(
            lower_bounds[candidate_index]
        )

        if lower_bound >= best_distance - 1e-12:
            break

        distance = exact_wasserstein_distance(
            query.signature,
            train_representations[
                int(candidate_index)
            ].signature,
        )
        emd_evaluations += 1

        if distance < best_distance:
            best_distance = distance
            best_index = int(candidate_index)

    if best_index < 0 or not np.isfinite(best_distance):
        raise RuntimeError(
            "Wasserstein 1-NN failed to find a neighbour."
        )

    return (
        int(train_labels[best_index]),
        best_index,
        float(best_distance),
        emd_evaluations,
    )


def run_experiment(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
) -> pd.DataFrame:
    train_representations = [
        represent_image(image) for image in train_images
    ]
    train_matrix = np.vstack(
        [item.flat for item in train_representations]
    )

    records: list[dict[str, object]] = []
    total_conditions = (
        len(test_images) * len(TRANSLATIONS)
    )
    completed_conditions = 0

    for test_id, (base_image, true_label) in enumerate(
        zip(test_images, test_labels)
    ):
        for shift_name, magnitude, dy, dx in TRANSLATIONS:
            shifted_image = translate_image(
                base_image,
                dy=dy,
                dx=dx,
            )
            query = represent_image(shifted_image)

            start = time.perf_counter()
            (
                euclidean_prediction,
                euclidean_neighbour,
                euclidean_distance,
            ) = euclidean_1nn(
                query.flat,
                train_matrix,
                train_labels,
            )
            euclidean_runtime = (
                time.perf_counter() - start
            )

            records.append(
                {
                    "test_id": test_id,
                    "true_label": int(true_label),
                    "shift_name": shift_name,
                    "magnitude": magnitude,
                    "dy": dy,
                    "dx": dx,
                    "metric": "Euclidean",
                    "prediction": euclidean_prediction,
                    "correct": int(
                        euclidean_prediction == true_label
                    ),
                    "neighbour_index": euclidean_neighbour,
                    "nearest_distance": euclidean_distance,
                    "runtime_seconds": euclidean_runtime,
                    "emd_evaluations": 0,
                }
            )

            start = time.perf_counter()
            (
                wasserstein_prediction,
                wasserstein_neighbour,
                wasserstein_distance,
                emd_evaluations,
            ) = wasserstein_1nn(
                query,
                train_representations,
                train_labels,
            )
            wasserstein_runtime = (
                time.perf_counter() - start
            )

            records.append(
                {
                    "test_id": test_id,
                    "true_label": int(true_label),
                    "shift_name": shift_name,
                    "magnitude": magnitude,
                    "dy": dy,
                    "dx": dx,
                    "metric": "1-Wasserstein",
                    "prediction": wasserstein_prediction,
                    "correct": int(
                        wasserstein_prediction == true_label
                    ),
                    "neighbour_index": wasserstein_neighbour,
                    "nearest_distance": wasserstein_distance,
                    "runtime_seconds": wasserstein_runtime,
                    "emd_evaluations": emd_evaluations,
                }
            )

            completed_conditions += 1
            if (
                completed_conditions
                % max(1, total_conditions // 20)
                == 0
            ):
                percentage = (
                    100 * completed_conditions
                    / total_conditions
                )
                print(
                    f"Progress: {percentage:5.1f}%",
                    flush=True,
                )

    return pd.DataFrame.from_records(records)


def add_neighbour_stability(
    results: pd.DataFrame,
) -> pd.DataFrame:
    clean = results.loc[
        results["magnitude"] == 0,
        ["test_id", "metric", "neighbour_index"],
    ].rename(
        columns={
            "neighbour_index": "clean_neighbour_index"
        }
    )

    merged = results.merge(
        clean,
        on=["test_id", "metric"],
        how="left",
    )
    merged["same_neighbour_as_clean"] = (
        merged["neighbour_index"]
        == merged["clean_neighbour_index"]
    ).astype(int)

    return merged


def summarize_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    summary = results.groupby(
        ["metric", "magnitude"],
        as_index=False,
    ).agg(
        observations=("correct", "size"),
        accuracy=("correct", "mean"),
        neighbour_stability=(
            "same_neighbour_as_clean", "mean"
        ),
        mean_runtime_seconds=(
            "runtime_seconds", "mean"
        ),
        total_runtime_seconds=(
            "runtime_seconds", "sum"
        ),
        mean_emd_evaluations=(
            "emd_evaluations", "mean"
        ),
    )

    clean_accuracy = summary.loc[
        summary["magnitude"] == 0,
        ["metric", "accuracy"],
    ].rename(
        columns={"accuracy": "clean_accuracy"}
    )

    summary = summary.merge(
        clean_accuracy,
        on="metric",
        how="left",
    )
    summary["robustness_loss"] = (
        summary["clean_accuracy"] - summary["accuracy"]
    )

    for column in [
        "accuracy",
        "neighbour_stability",
        "clean_accuracy",
        "robustness_loss",
    ]:
        summary[column] *= 100.0

    return summary


def paired_bootstrap_accuracy_difference(
    results: pd.DataFrame,
    magnitude: int,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    subset = results.loc[
        results["magnitude"] == magnitude
    ]
    per_image_metric = subset.groupby(
        ["test_id", "metric"],
        as_index=False,
    )["correct"].mean()

    pivot = per_image_metric.pivot(
        index="test_id",
        columns="metric",
        values="correct",
    ).dropna()

    if (
        "Euclidean" not in pivot
        or "1-Wasserstein" not in pivot
    ):
        raise ValueError(
            "Both metrics are required "
            "for paired bootstrapping."
        )

    differences = (
        pivot["1-Wasserstein"].to_numpy()
        - pivot["Euclidean"].to_numpy()
    )

    rng = np.random.default_rng(seed + magnitude)
    n_images = len(differences)
    bootstrap_means = np.empty(
        repetitions,
        dtype=np.float64,
    )

    for repetition in range(repetitions):
        sampled_indices = rng.integers(
            0,
            n_images,
            size=n_images,
        )
        bootstrap_means[repetition] = (
            differences[sampled_indices].mean()
        )

    estimate = float(differences.mean())
    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    return {
        "magnitude": magnitude,
        "test_images": n_images,
        "accuracy_difference_percentage_points": (
            100.0 * estimate
        ),
        "ci_95_lower": 100.0 * float(lower),
        "ci_95_upper": 100.0 * float(upper),
    }


def calculate_bootstrap_table(
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        paired_bootstrap_accuracy_difference(
            results,
            magnitude=magnitude,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=RANDOM_SEED,
        )
        for magnitude in sorted(
            results["magnitude"].unique()
        )
    ]
    return pd.DataFrame(rows)


def save_accuracy_plot(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.figure(figsize=(7, 5))

    for metric, group in summary.groupby("metric"):
        group = group.sort_values("magnitude")
        plt.plot(
            group["magnitude"],
            group["accuracy"],
            marker="o",
            label=metric,
        )

    plt.xlabel("Translation magnitude (pixels)")
    plt.ylabel("Classification accuracy (%)")
    plt.title("1-NN Accuracy Under Image Translation")
    plt.xticks(sorted(summary["magnitude"].unique()))
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "accuracy_vs_translation.png",
        dpi=300,
    )
    plt.close()


def save_robustness_loss_plot(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    shifted = summary.loc[
        summary["magnitude"] > 0
    ].copy()

    pivot = shifted.pivot(
        index="magnitude",
        columns="metric",
        values="robustness_loss",
    )

    ax = pivot.plot(kind="bar", figsize=(7, 5))
    ax.set_xlabel("Translation magnitude (pixels)")
    ax.set_ylabel("Accuracy loss (percentage points)")
    ax.set_title("Translation-Induced Accuracy Loss")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(
        output_dir / "robustness_loss.png",
        dpi=300,
    )
    plt.close()


def save_neighbour_stability_plot(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    shifted = summary.loc[summary["magnitude"] > 0]
    plt.figure(figsize=(7, 5))

    for metric, group in shifted.groupby("metric"):
        group = group.sort_values("magnitude")
        plt.plot(
            group["magnitude"],
            group["neighbour_stability"],
            marker="o",
            label=metric,
        )

    plt.xlabel("Translation magnitude (pixels)")
    plt.ylabel(
        "Same nearest neighbour as clean image (%)"
    )
    plt.title(
        "Nearest-Neighbour Stability Under Translation"
    )
    plt.xticks(sorted(shifted["magnitude"].unique()))
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_dir / "neighbour_stability.png",
        dpi=300,
    )
    plt.close()


def save_runtime_plot(
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    runtime = summary.groupby(
        "metric",
        as_index=False,
    )["mean_runtime_seconds"].mean()

    plt.figure(figsize=(6, 5))
    plt.bar(
        runtime["metric"],
        runtime["mean_runtime_seconds"],
    )
    plt.ylabel(
        "Mean classification time per image (seconds)"
    )
    plt.title("Computational Cost of Each Distance Metric")
    plt.tight_layout()
    plt.savefig(
        output_dir / "runtime_comparison.png",
        dpi=300,
    )
    plt.close()


def save_sample_translation_plot(
    original_image: np.ndarray,
    output_dir: Path,
) -> None:
    examples = [
        ("Original", original_image),
        (
            "Right by 1",
            translate_image(original_image, 0, 1),
        ),
        (
            "Down by 1",
            translate_image(original_image, 1, 0),
        ),
        (
            "Right by 2",
            translate_image(original_image, 0, 2),
        ),
    ]

    fig, axes = plt.subplots(
        1,
        len(examples),
        figsize=(10, 3),
    )

    for axis, (title, image) in zip(axes, examples):
        axis.imshow(image, cmap="gray_r")
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(
        output_dir / "sample_translations.png",
        dpi=300,
    )
    plt.close(fig)


def save_confusion_matrices(
    results: pd.DataFrame,
    output_dir: Path,
) -> None:
    largest_magnitude = int(
        results["magnitude"].max()
    )
    subset = results.loc[
        results["magnitude"] == largest_magnitude
    ]

    for metric, group in subset.groupby("metric"):
        matrix = confusion_matrix(
            group["true_label"],
            group["prediction"],
            labels=np.arange(10),
        )

        plt.figure(figsize=(7, 6))
        plt.imshow(matrix)
        plt.colorbar()
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(10))
        plt.xlabel("Predicted digit")
        plt.ylabel("True digit")
        plt.title(
            f"{metric}: Confusion Matrix at "
            f"{largest_magnitude}-Pixel Shift"
        )

        for row in range(10):
            for column in range(10):
                plt.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                )

        plt.tight_layout()
        safe_metric = (
            metric.lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        plt.savefig(
            output_dir
            / f"confusion_matrix_{safe_metric}.png",
            dpi=300,
        )
        plt.close()


def print_key_results(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    display_columns = [
        "metric",
        "magnitude",
        "accuracy",
        "robustness_loss",
        "neighbour_stability",
        "mean_runtime_seconds",
    ]

    print("\nMAIN RESULTS")
    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nPAIRED BOOTSTRAP: WASSERSTEIN ACCURACY "
        "MINUS EUCLIDEAN ACCURACY"
    )
    print(
        bootstrap.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    clean = summary.loc[
        summary["magnitude"] == 0
    ].set_index("metric")

    largest_shift = int(summary["magnitude"].max())
    shifted = summary.loc[
        summary["magnitude"] == largest_shift
    ].set_index("metric")

    euclidean_time = summary.loc[
        summary["metric"] == "Euclidean",
        "mean_runtime_seconds",
    ].mean()

    wasserstein_time = summary.loc[
        summary["metric"] == "1-Wasserstein",
        "mean_runtime_seconds",
    ].mean()

    runtime_ratio = (
        wasserstein_time / euclidean_time
        if euclidean_time > 0
        else np.nan
    )

    print("\nABSTRACT VALUES")
    print(f"Training images: {10 * TRAIN_PER_CLASS}")
    print(f"Test images: {10 * TEST_PER_CLASS}")
    print(
        "Euclidean clean accuracy: "
        f"{clean.loc['Euclidean', 'accuracy']:.2f}%"
    )
    print(
        "Wasserstein clean accuracy: "
        f"{clean.loc['1-Wasserstein', 'accuracy']:.2f}%"
    )
    print(
        f"Euclidean accuracy at {largest_shift} pixels: "
        f"{shifted.loc['Euclidean', 'accuracy']:.2f}%"
    )
    print(
        f"Wasserstein accuracy at {largest_shift} pixels: "
        f"{shifted.loc['1-Wasserstein', 'accuracy']:.2f}%"
    )
    print(
        f"Euclidean robustness loss at {largest_shift} "
        "pixels: "
        f"{shifted.loc['Euclidean', 'robustness_loss']:.2f}"
        " percentage points"
    )
    print(
        f"Wasserstein robustness loss at {largest_shift} "
        "pixels: "
        f"{shifted.loc['1-Wasserstein', 'robustness_loss']:.2f}"
        " percentage points"
    )
    print(
        "Approximate runtime ratio W1/Euclidean: "
        f"{runtime_ratio:.1f}x"
    )


def main() -> int:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise ValueError(
            "Output directory is nonempty; "
            "choose a new OUTPUT_DIR."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    versions = {}
    for package in [
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "opencv-python",
    ]:
        try:
            versions[package] = (
                importlib.metadata.version(package)
            )
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    metadata = dict(
        seed=RANDOM_SEED,
        train_per_class=TRAIN_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
        canvas_size=CANVAS_SIZE,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        bootstrap_method=(
            "Paired image-level percentile interval; "
            "seed + magnitude"
        ),
        translations=TRANSLATIONS,
        quick_mode=QUICK_MODE,
        python=sys.version,
        platform=platform.platform(),
        versions=versions,
        transport=(
            "OpenCV EMD with Euclidean ground cost "
            "and centroid pruning"
        ),
        timing=(
            "One classification call per query and "
            "condition; excludes image preparation"
        ),
    )
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("Loading the handwritten-digits dataset...")
    digits = load_digits()
    raw_images = digits.images.astype(np.float64)
    labels = digits.target.astype(int)

    (
        raw_train_images,
        train_labels,
        raw_test_images,
        test_labels,
    ) = choose_balanced_samples(
        raw_images,
        labels,
        train_per_class=TRAIN_PER_CLASS,
        test_per_class=TEST_PER_CLASS,
        seed=RANDOM_SEED,
    )

    train_images = np.stack([
        pad_image(image, CANVAS_SIZE)
        for image in raw_train_images
    ])
    test_images = np.stack([
        pad_image(image, CANVAS_SIZE)
        for image in raw_test_images
    ])

    print(f"Training images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")
    print(f"Translation conditions: {len(TRANSLATIONS)}")
    print(f"Quick mode: {QUICK_MODE}")
    print(f"Results folder: {OUTPUT_DIR}")

    save_sample_translation_plot(
        test_images[0],
        OUTPUT_DIR,
    )

    experiment_start = time.perf_counter()
    results = run_experiment(
        train_images,
        train_labels,
        test_images,
        test_labels,
    )
    elapsed = time.perf_counter() - experiment_start

    results = add_neighbour_stability(results)
    summary = summarize_results(results)
    bootstrap = calculate_bootstrap_table(results)

    results.to_csv(
        OUTPUT_DIR / "detailed_predictions.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_DIR / "summary_results.csv",
        index=False,
    )
    bootstrap.to_csv(
        OUTPUT_DIR / "bootstrap_accuracy_differences.csv",
        index=False,
    )

    save_accuracy_plot(summary, OUTPUT_DIR)
    save_robustness_loss_plot(summary, OUTPUT_DIR)
    save_neighbour_stability_plot(summary, OUTPUT_DIR)
    save_runtime_plot(summary, OUTPUT_DIR)
    save_confusion_matrices(results, OUTPUT_DIR)

    print_key_results(summary, bootstrap)
    print(f"\nTotal experiment time: {elapsed:.2f} seconds")
    print(f"All files were saved to: {OUTPUT_DIR}")

    if QUICK_MODE:
        print(
            "\nWARNING: QUICK_MODE is enabled. "
            "These results are only for checking "
            "the program. Set QUICK_MODE = False "
            "before producing the final paper results."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nExperiment stopped by the user.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
