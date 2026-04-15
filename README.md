# Von Neumann Entropy of Intertidal Community Organization

Analysis code and data for:

**Rodriguez AM (2026).** Von Neumann entropy reveals alternative stable state architecture in intertidal community succession. Submitted to *Oecologia*.

## What this does

Computes Von Neumann entropy of species correlation density matrices to quantify the effective dimensionality of community interaction structure. Applied to Petraitis and colleagues' 1996–2002 clearing experiment on Swan's Island, Gulf of Maine.

The main result: Von Neumann entropy declines over succession (interaction network self-organizes toward lower-dimensional structure) and varies non-monotonically with clearing size, with 2 m² clearings producing the most organized community state. Shannon entropy does not reliably detect either pattern.

## Files

- `intertidal_vn_entropy.py` — Full analysis: Von Neumann entropy computation, community detection, bootstrap CIs, species-label and row-label permutation tests, all figures. Runtime ~28 min.
- `shannon_comparison.py` — Shannon H' and Simpson's D on the same strata for comparison. Runtime ~seconds.
- `Succession_sampling_96-02_data.txt` — Raw data from Petraitis and Vidargas (2006), unmodified. Originally published as Ecological Archives E087-047-D1. See Data section below.
- `Rodriguez_VN_Entropy_Oecologia_submitted.pdf` — Manuscript preprint, submitted to Oecologia for review, April 2026.

## Usage

```
python intertidal_vn_entropy.py
python shannon_comparison.py
```

Both scripts expect `Succession_sampling_96-02_data.txt` in the working directory. Output goes to `intertidal_vn_output/`.

Requires Python 3 with NumPy, SciPy, pandas, and matplotlib.

## Data

The dataset is from:

Petraitis PS, Vidargas N (2006). Marine intertidal organisms found in experimental clearings on sheltered shores, Gulf of Maine, USA. *Ecology* 87:796. [doi:10.1890/05-0232a](https://doi.org/10.1890/05-0232a)

Also available from Figshare: [doi:10.6084/m9.figshare.3526004](https://doi.org/10.6084/m9.figshare.3526004)

The original metadata states: "Copyright restrictions: None. Proprietary restrictions: None." The author of this study thanks the creators of the original dataset for their work generating the data, and their commitment to open science by their sharing of it.

## License

Code is MIT licensed. See LICENSE file. The data file retains its original terms as published by Petraitis and Vidargas (2006).
