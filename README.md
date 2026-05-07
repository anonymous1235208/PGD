# PGD

Research code for kernel-based optimization and testing experiments built with JAX. This repository contains standalone experiment scripts for:

- MMD gradient flow on a Gaussian mixture target
- Parameter inference for the `g-and-k` distribution
- Parameter inference for a stochastic Lotka-Volterra model
- Goodness-of-fit testing for a toggle-switch model using parametric bootstrap

The code is organized as experiment scripts rather than a packaged library. Most runs save compressed `.npz` result files plus a small JSON metadata sidecar.

## Repository Layout

- [jnp_main.py](/Users/sophiakang/Documents/GitHub/PGD/jnp_main.py): MMD gradient-flow experiment on an 8-component Gaussian mixture
- [g_and_k.py](/Users/sophiakang/Documents/GitHub/PGD/g_and_k.py): `g-and-k` parameter inference with SGD, natural SGD, and PGD-style updates
- [lotkka_volterra.py](/Users/sophiakang/Documents/GitHub/PGD/lotkka_volterra.py): stochastic Lotka-Volterra inference experiments
- [ts.py](/Users/sophiakang/Documents/GitHub/PGD/ts.py): toggle-switch goodness-of-fit experiments with bootstrap testing
- [utils.py](/Users/sophiakang/Documents/GitHub/PGD/utils.py): figure-generation utilities for saved experiment outputs
- [results_io.py](/Users/sophiakang/Documents/GitHub/PGD/results_io.py): helpers for saving metadata sidecars
- [composite_tests](/Users/sophiakang/Documents/GitHub/PGD/composite_tests): supporting kernels, bootstrap tests, estimators, and distributions


## How To Run

### 1. MMD Gradient Flow

`jnp_main.py` runs the particle-flow experiment and writes a result archive such as `results_n10f.npz`.

```bash
python jnp_main.py
```

Key settings are currently configured near the bottom of the file inside the `if __name__ == "__main__":` block.

### 2. `g-and-k` Inference

`g_and_k.py` runs single experiments and several ablation/grid modes. Choose the mode by editing `experiment_mode` near the bottom of the file.

```bash
python g_and_k.py
```

Common modes in the script:

- `single_run`
- `step_size_ablation`
- `decay_ablation`
- `observation_model_grid`
- `lengthscale_ablation`
- `regularization_ablation`
- `lengthscale_regularization_grid`

### 3. Lotka-Volterra Inference

`lotkka_volterra.py` follows the same pattern as `g_and_k.py`: edit `experiment_mode` and the experiment hyperparameters at the bottom of the file, then run:

```bash
python lotkka_volterra.py
```

Outputs are typically written under `results/` or `ablations/`, depending on the selected mode.

### 4. Toggle-Switch Goodness-of-Fit

`ts.py` has a command-line interface. Example:

```bash
python ts.py --Ts 10 --methods pgd sgd --seeds 160 --bootstrap 100 --output results/toggle_switch.csv
```

Useful arguments:

- `--Ts`: list of experiment horizons
- `--methods`: `pgd` and/or `sgd`
- `--n`: sample size
- `--bootstrap`: number of bootstrap replicates
- `--iterations`: optimization iterations per fit

### 5. Regenerate Figures

`utils.py` rebuilds figures from saved result files.

```bash
python utils.py all
```

You can also target one group of figures at a time:

```bash
python utils.py gnk
python utils.py lv
python utils.py mmd_flow
```

## Outputs

Most experiment scripts save:

- a compressed NumPy archive (`.npz`) with arrays and summary statistics
- a metadata sidecar (`*_meta.json`) containing small, JSON-friendly run details

The sidecar-writing logic lives in [results_io.py](/Users/sophiakang/Documents/GitHub/PGD/results_io.py).
