# Radial simulations of breast cancer aerotaxis

This repository contains Python solvers and plotting scripts for a radial model of breast cancer cell migration along oxygen gradients.

- `full_reduced/` compares the phenotype-resolved model with its reduced approximation.
- `reduced_controls/` uses the reduced model to examine the effects of local growth, respiration, receptor activation and initial cluster size.

## Usage

From the directory containing this README, install the dependencies and run the simulations.

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The script checks the computed profiles against the supplied reference data. It saves the figures in `outputs/figures/` and the profiles in `outputs/profiles/`.
