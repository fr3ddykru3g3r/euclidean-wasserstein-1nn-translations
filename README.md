# Euclidean and Wasserstein 1-NN Classification Under Image Translations

Code for the study by Vinamra Dhoot and Kyros Goyal.

The paper compares Euclidean and exact 1-Wasserstein distances in a 1-nearest-neighbour classifier on scikit-learn's handwritten-digits dataset. It also evaluates translation-optimized Euclidean (TOE) comparison and a sufficient certificate for robustness to bounded integer translations.

## Contents

- `baseline_experiment.py` — baseline Euclidean and Wasserstein comparison, paired bootstrap summaries, and figures (Appendix A, Listing 1).
- `translation_optimized.py` — TOE comparison, exact Wasserstein comparison, and translation-certificate evaluation (Appendix A, Listing 2).
- `requirements.txt` — dependencies for both scripts.
- `CITATION.cff` — citation metadata for the associated manuscript.

The scripts use `sklearn.datasets.load_digits`; no separate dataset download is required. The baseline script saves the stratified split to `wasserstein_1nn_results/split_indices.npz`, which the second script can reuse to keep the split identical.

## Requirements

Python 3.9 or later is required. Create an environment and install the listed packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the experiments

Run the baseline comparison:

```bash
python baseline_experiment.py
```

This writes predictions, summary tables, split indices, metadata, and figures to `wasserstein_1nn_results/`. The script stops if that directory already contains files; move or rename the existing output directory before rerunning.

Run TOE and its certificate check while skipping the additional Wasserstein calculation:

```bash
python translation_optimized.py \
  --split wasserstein_1nn_results/split_indices.npz \
  --skip-wasserstein \
  --out toe_results
```

To include exact Wasserstein in the second experiment, omit `--skip-wasserstein` and choose an empty output directory:

```bash
python translation_optimized.py \
  --split wasserstein_1nn_results/split_indices.npz \
  --out toe_results_with_wasserstein
```

At the default sample sizes, the second script reports 405,000 exact transport problems for its Wasserstein comparison, so that part can take a long time. Both scripts refuse to write into a nonempty results directory.

## Reported results

These are the values reported in the manuscript.

| Method | Untranslated | 1-pixel shifts | 2-pixel shifts |
| --- | ---: | ---: | ---: |
| Euclidean 1-NN | 98.00% | 56.67% | 11.67% |
| 1-Wasserstein 1-NN | 92.67% | 42.33% | 15.83% |
| TOE, radius 2 | 97.33% | 97.33% | 97.50% |

The manuscript also reports that the radius-2 sufficient condition certified 146 of 150 test images, and all 146 were correct under all 25 translations in the radius-2 square.

## Source note

The Python files are transcriptions of the source listings in Appendix A. A handful of curly apostrophes in dictionary and index expressions were changed to ASCII single quotes so the listings parse as Python. No experiment outputs are bundled; running the scripts creates them locally.

## License

No reuse license has been selected for this code. Public repository visibility does not itself grant permission to redistribute or adapt it.

## References

- T. M. Cover and P. E. Hart, “Nearest neighbor pattern classification,” *IEEE Transactions on Information Theory*, 13(1), 21–27, 1967.
- Y. Rubner, C. Tomasi, and L. J. Guibas, “The earth mover's distance as a metric for image retrieval,” *International Journal of Computer Vision*, 40(2), 99–121, 2000.
- P. Simard, Y. LeCun, and J. Denker, “Efficient pattern recognition using a new transformation distance,” *Advances in Neural Information Processing Systems*, 5, 1993.
- scikit-learn, [`load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
