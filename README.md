# Masked Multiscale Image Energies — code and numerical records

Version 1.0.0 (code/data release). This repository accompanies research on **masked multiscale image energies and Haar-scale statistics**. It distributes **software, selected fixed image inputs, and saved numerical results only**. **No manuscript, preprint, article PDF, LaTeX source, or scholarly supplement is included.**

## Authors
Sergo A. Episkoposian; Svetlana Artur Grigoryan; Davit Samvel Hovhannisyan; Spartak Sevak Sirakanyan (order preserved from source package).

## Repository contents
- `code/mask_profiles.py`: Haar energy profiles, masked moments, and rank-testing primitives.
- `code/`: simulation and experimental scripts; some scripts are retained from research workflows and may refer to omitted publication-specific outputs.
- `results/`: saved CSV/NPZ numerical data and verification records (no typeset paper tables).
- `data/new_templates/`: photographic input and provenance/rights information.
- `landsat_pilot/`: fixed raster test inputs, numerical pilot data, scripts and source/rights documentation.
- `figures/`: excluded (manuscript-format PDF figures were deliberately removed); PNG pilot plots are retained in `landsat_pilot/figures/`.

## Installation
Python 3.12 was used for the source research package. Install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Basic smoke test
From the repository root:

```bash
python -c "import sys; sys.path.insert(0, 'code'); import numpy as np; from mask_profiles import profiles; print(profiles(np.arange(16, dtype=float).reshape(4,4)))"
```

To explore experiments, inspect `code/run_experiments.py`, `code/unified_experiments.py`, and `landsat_pilot/code/run_pilot.py` and their saved configuration files. **Run stochastic experiments in a separate working copy:** some scripts overwrite saved outputs. The source package's `make verify` and other manuscript-closure checks are not supplied as verified commands because they depended on excluded TeX documents. Numerical evidence is retained, but the full research pipeline has **not been rerun or independently revalidated** for this manuscript-free subset.

## Scope and limitations
Results concern controlled-noise experiments using fixed input templates; they do not establish clinical performance or population-wide imaging validity. Some saved logs or configurations may reference a manuscript or original research review workflow; these references are historical, not included publication files. The code/data release does not imply journal acceptance or an article DOI.

## Licensing and attribution
**No general code license is granted in this preliminary public-release candidate.** Obtain approval from all relevant code rightsholders and add an appropriate license (e.g. MIT or BSD-3-Clause) before publication. Third-party inputs retain their own terms; consult `THIRD_PARTY_NOTICES.md`, `data/new_templates/SOURCE_AND_RIGHTS.md` and `landsat_pilot/data/RASTERIO_DATA_README.rst`. Do not claim original-code licensing rights for third-party assets.

## Citation and archiving
Use `CITATION.cff` for authorship and title. After creating a GitHub release, archive that release with Zenodo and add the actual version DOI and repository URL to this README and the citation metadata. No DOI has been assigned in this file.
