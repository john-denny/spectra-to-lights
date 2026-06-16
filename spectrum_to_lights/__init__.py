"""spectrum_to_lights — spectral matching for a 24-channel LED array."""

import pathlib
import webbrowser

import numpy as np

from ._lib import (
    WAVELENGTHS,
    load_matrix,
    load_target,
    match_quality,
    save_plot,
    seed_from_spectrum,
)
from ._solvers import DEFAULT_SOLVER, SOLVERS, get_solver

__all__ = ["SpectralSolver"]


class SpectralSolver:
    """Matches a target spectrum to 24-channel DMX values.

    Parameters
    ----------
    solver:
        Default solver name. One of: lsq_linear, nnls, lasso, ridge,
        elastic_net, diffevo, weighted_ls, pgd, admm, dual_annealing, huber,
        or "all" to run every solver and return the best result by SAM.
    """

    def __init__(self, solver: str = DEFAULT_SOLVER) -> None:
        self._A = load_matrix()
        self._default_solver = solver

    def _load(self, target: "str | pathlib.Path | np.ndarray") -> np.ndarray:
        return load_target(str(target)) if not isinstance(target, np.ndarray) else target

    def _run(self, target: "str | pathlib.Path | np.ndarray", solver_name: str | None) -> tuple[np.ndarray, np.ndarray]:
        t = self._load(target)
        if (solver_name or self._default_solver) == "all":
            return t, self._best(t)
        x0 = seed_from_spectrum(t)
        x = get_solver(solver_name or self._default_solver).solve(self._A, t, x0)
        return t, np.clip(x, 0, 1)

    def _best(self, t: np.ndarray) -> np.ndarray:
        x0 = seed_from_spectrum(t)
        best_x, best_sam = None, float("inf")
        for solver in SOLVERS.values():
            x = np.clip(solver.solve(self._A, t, x0), 0, 1)
            sam, _, _ = match_quality(self._A, x, t)
            if sam < best_sam:
                best_sam, best_x = sam, x
        return best_x

    def solve(
        self,
        target: "str | pathlib.Path | np.ndarray",
        *,
        solver: str | None = None,
        brightness: float = 1.0,
    ) -> np.ndarray:
        """Return 24 DMX values (integers 0–255).

        Parameters
        ----------
        target:
            Path to a CSV with (wavelength, intensity) columns, or a
            pre-loaded array of shape (401,) on the 380–780 nm grid.
        solver:
            Override the instance default solver for this call.
        brightness:
            Scale factor applied before converting to DMX range (0.0–1.0).
        """
        _, x = self._run(target, solver)
        return np.round(x * brightness * 255).astype(int)

    def quality(
        self,
        target: "str | pathlib.Path | np.ndarray",
        *,
        solver: str | None = None,
    ) -> tuple[float, float, float]:
        """Return (sam_degrees, r2, rmse) match quality metrics.

        Lower SAM and RMSE, higher R² = better match.
        """
        t, x = self._run(target, solver)
        return match_quality(self._A, x, t)

    def plot(
        self,
        target: "str | pathlib.Path | np.ndarray",
        *,
        solver: str | None = None,
        out: str = "match.html",
        open_browser: bool = True,
    ) -> None:
        """Save a Plotly HTML plot of target vs reconstructed spectrum.

        Parameters
        ----------
        target:
            Path to a CSV or pre-loaded (401,) array.
        solver:
            Override the instance default solver for this call.
        out:
            Output HTML file path.
        open_browser:
            If True, open the saved file in the default browser.
        """
        t, x = self._run(target, solver)
        reconstructed = self._A @ x
        sam, r2, rmse = match_quality(self._A, x, t)
        save_plot(WAVELENGTHS, t, reconstructed, sam, r2, rmse, out)
        if open_browser:
            webbrowser.open(pathlib.Path(out).resolve().as_uri())
