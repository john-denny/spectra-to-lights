# spectra-to-lights
A Python package for spectral matching on the Colordyne tuneable spectrum lights in the imaging lab at University of Galway.

## Our setup
The Colordyne rig uses 24 individually addressable LEDs, each with a distinct spectral power distribution (SPD) spanning the visible range and into NIR. Given a target spectrum — such as a camera QE curve or a standard illuminant — this package finds the 24 DMX channel values (0–255) that make the rig best reproduce that spectrum.

The SPD data for all 24 channels is bundled with the package, so no external data files are needed.

## Install

```
uv add git+https://github.com/john-denny/spectra-to-lights
```
Or with pip:
```
pip install git+https://github.com/john-denny/spectra-to-lights
```

## Usage

```python
from spectrum_to_lights import SpectralSolver

solver = SpectralSolver()

# Get 24 DMX values (0–255) that best reproduce the target spectrum
dmx = solver.solve("QE_green.csv")

# Check how well the match worked
sam, r2, rmse = solver.quality("QE_green.csv")
print(f"SAM={sam:.1f}°  R²={r2:.3f}  RMSE={rmse:.4f}")

# Save and open a Plotly HTML plot of target vs reconstructed spectrum
solver.plot("QE_green.csv", out="match.html")
```

The target CSV should have two columns: wavelength (nm) and intensity. Values are interpolated onto the internal 380–780 nm grid automatically.

You can also pass a pre-loaded NumPy array (shape `(401,)`) instead of a file path.

### Solvers

The default solver is `lsq_linear` — scipy's bounded least-squares. Eleven solvers are available:

| Name | Method |
|---|---|
| `lsq_linear` | Bounded least-squares (default) |
| `nnls` | Non-negative least-squares |
| `lasso` | Lasso (L1 regularisation, sparse) |
| `ridge` | Tikhonov (L2 regularisation) |
| `elastic_net` | ElasticNet (L1 + L2) |
| `diffevo` | Differential evolution (global) |
| `weighted_ls` | Gaussian-weighted LS (peaks at 555 nm) |
| `pgd` | FISTA accelerated projected gradient descent |
| `admm` | Alternating direction method of multipliers |
| `dual_annealing` | Dual annealing (global) |
| `huber` | IRLS with Huber loss (robust) |

Override the solver per-call:

```python
dmx = solver.solve("QE_green.csv", solver="huber")
solver.plot("QE_green.csv", solver="weighted_ls", out="match.html")
```

Apply a brightness scale before converting to DMX range:

```python
dmx = solver.solve("QE_green.csv", brightness=0.8)
```

### Sending to hardware

The DMX values returned by `solve()` can be passed directly to [colordyne-controller](https://github.com/john-denny/colordyne-controller):

```python
from spectrum_to_lights import SpectralSolver
from vm116 import VM116

solver = SpectralSolver()
dmx = solver.solve("QE_green.csv")

with VM116() as dmx_out:
    for ch, val in enumerate(dmx, start=1):
        dmx_out.set_channel(ch, int(val))
    dmx_out.send()
```

## Requirements

- Python 3.11+
- numpy, pandas, scipy, scikit-learn, plotly (installed automatically)
- For hardware output: [colordyne-controller](https://github.com/john-denny/colordyne-controller)

## Channel index

Current configuration of the Colordyne rig in the imaging lab at University of Galway.

| DMX Channel | Name | Peak (nm) |
|---|---|---|
| **1** | lime | 533 |
| **2** | RYL B40 | 447 |
| **3** | Violet421 | 422 |
| **4** | lime | 530 |
| **5** | DR-660 | 660 |
| **6** | Violet405 | 404 |
| **7** | pc-Amber | 597 |
| **8** | DR-735 | 735 |
| **9** | red orange | 617 |
| **10** | lime | 535 |
| **11** | DR-700 | 694 |
| **12** | Sky Blue | 473 |
| **13** | NIR-940 | 942 |
| **14** | DR-680 | 677 |
| **15** | lime | 530 |
| **16** | pc-Amber | 596 |
| **17** | cyan 40 | 494 |
| **18** | lime | 533 |
| **19** | FR-850 | 851 |
| **20** | OS-640 | 636 |
| **21** | lime | 533 |
| **22** | green | 523 |
| **23** | amber | 593 |
| **24** | lime | 531 |

Channels 13 (942 nm) and 19 (851 nm) are NIR and fall outside the 380–780 nm solver grid — they are always set to 0.
