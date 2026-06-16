"""Solver implementations for spectral matching (minimise ||Ax - t||² s.t. 0 ≤ x ≤ 1)."""

from typing import Protocol

import numpy as np
from scipy.optimize import differential_evolution, lsq_linear, nnls

# 401-point wavelength axis — matches the rows of the SPD matrix loaded by lib.py
_WL = np.arange(380, 781, 1)


class Solver(Protocol):
    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Original solvers
# ---------------------------------------------------------------------------

class LsqLinearSolver:
    """Bounded least-squares via scipy.optimize.lsq_linear (default)."""

    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        result = lsq_linear(A, t, bounds=(0, 1))
        return np.clip(result.x, 0, 1)


class NNLSSolver:
    """Non-negative least-squares; clips result to [0, 1]."""

    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        x, _ = nnls(A, t)
        return np.clip(x, 0, 1)


class LassoSolver:
    """Lasso regression (positive=True) — promotes sparse channel use."""

    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        from sklearn.linear_model import Lasso
        model = Lasso(alpha=0.001, positive=True, max_iter=10000)
        model.fit(A, t)
        return np.clip(model.coef_, 0, 1)


class RidgeSolver:
    """Tikhonov (L2) regularisation via augmented lsq_linear — respects [0,1] bounds."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        n = A.shape[1]
        A_aug = np.vstack([A, np.sqrt(self.alpha) * np.eye(n)])
        t_aug = np.concatenate([t, np.zeros(n)])
        result = lsq_linear(A_aug, t_aug, bounds=(0, 1))
        return np.clip(result.x, 0, 1)


class ElasticNetSolver:
    """ElasticNet (positive=True) — balance of sparse and smooth."""

    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(alpha=0.001, l1_ratio=0.5, positive=True, max_iter=10000)
        model.fit(A, t)
        return np.clip(model.coef_, 0, 1)


class DiffEvoSolver:
    """Differential evolution global optimiser — no local-minimum risk."""

    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        result = differential_evolution(
            lambda x: float(np.dot(A @ x - t, A @ x - t)),
            bounds=[(0, 1)] * A.shape[1],
            seed=42,
            maxiter=300,
            popsize=12,
            tol=1e-6,
            workers=1,
        )
        return np.clip(result.x, 0, 1)


# ---------------------------------------------------------------------------
# New solvers
# ---------------------------------------------------------------------------

class WeightedLSSolver:
    """Spectrally weighted LS — Gaussian peaking at 555 nm de-emphasises UV/NIR.

    Changes the objective to ||W(Ax-t)||² so errors in the visible-range peak
    matter most. Channels 13 (942 nm) and 19 (851 nm) lie outside the visible
    band and can inflate the unweighted residual; this suppresses their influence.
    """

    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        w = np.exp(-((_WL - 555.0) ** 2) / (2 * 120.0 ** 2))
        result = lsq_linear(A * w[:, None], t * w, bounds=(0, 1))
        return np.clip(result.x, 0, 1)


class PGDSolver:
    """FISTA — Nesterov-accelerated projected gradient descent.

    Iterates: y ← x + momentum*(x - x_prev); x ← clip(y - step*∇f(y), 0, 1).
    Avoids direct linear-system solves, which can amplify ill-conditioning in A
    when LED spectra are highly correlated (e.g. the four ~530 nm lime channels).
    Seeded from the QE-lookup initial guess when x0 is provided.
    """

    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        ATA = A.T @ A
        ATt = A.T @ t
        L = float(np.linalg.norm(ATA, ord=2))  # Lipschitz constant of ∇f
        step = 1.0 / L

        n = A.shape[1]
        x = np.clip(x0.copy(), 0, 1) if x0 is not None else np.full(n, 0.5)
        y = x.copy()
        k = 1.0

        for _ in range(3000):
            grad = ATA @ y - ATt
            x_new = np.clip(y - step * grad, 0, 1)
            k_new = (1.0 + np.sqrt(1.0 + 4.0 * k * k)) / 2.0
            y = x_new + ((k - 1.0) / k_new) * (x_new - x)
            x, k = x_new, k_new

        return np.clip(x, 0, 1)


class ADMMSolver:
    """ADMM — alternating direction method of multipliers with box projection.

    Decouples the LS solve (x-update via pre-factored Cholesky of AᵀA + ρI)
    from the [0,1] constraint (z-update = elementwise clip). The two simple
    sub-problems are cheaper and more stable than a single constrained solve.
    Seeded from the QE-lookup initial guess when x0 is provided.
    """

    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        n = A.shape[1]
        rho = 1.0
        ATA = A.T @ A
        ATt = A.T @ t
        L_chol = np.linalg.cholesky(ATA + rho * np.eye(n))

        x = np.clip(x0.copy(), 0, 1) if x0 is not None else np.zeros(n)
        z = x.copy()
        u = np.zeros(n)

        for _ in range(1000):
            rhs = ATt + rho * (z - u)
            x = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, rhs))
            z = np.clip(x + u, 0, 1)
            u = u + x - z

        return np.clip(z, 0, 1)


class DualAnnealingSolver:
    """Dual annealing — single-trajectory SA with local gradient polishing.

    Unlike differential evolution's population search, dual annealing walks a
    single trajectory with temperature-driven jumps and periodically descends
    to the nearest local minimum via L-BFGS-B. Often finds better solutions
    on smooth but multimodal loss surfaces.
    Seeded from the QE-lookup initial guess when x0 is provided.
    """

    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        from scipy.optimize import dual_annealing
        n = A.shape[1]
        result = dual_annealing(
            lambda x: float(np.dot(A @ x - t, A @ x - t)),
            bounds=[(0.0, 1.0)] * n,
            seed=42,
            maxiter=1000,
            x0=x0,
            minimizer_kwargs={"method": "L-BFGS-B", "bounds": [(0.0, 1.0)] * n},
        )
        return np.clip(result.x, 0, 1)


class HuberSolver:
    """IRLS with Huber loss — robust to spectral bands the LED set cannot reach.

    L2 wastes coefficient budget matching regions with zero LED coverage (large
    unavoidable residuals). Huber down-weights those residuals past a threshold δ,
    so subsequent iterations focus where the LEDs can actually match the target.
    """

    def solve(self, A: np.ndarray, t: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        delta = max(0.05 * float(np.std(t)), 1e-6)
        x = np.clip(lsq_linear(A, t, bounds=(0, 1)).x, 0, 1)  # lsq warm start beats QE seed

        for _ in range(30):
            r = np.abs(A @ x - t)
            w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
            x_new = np.clip(lsq_linear(A * w[:, None], t * w, bounds=(0, 1)).x, 0, 1)
            if np.linalg.norm(x_new - x) < 1e-9:
                break
            x = x_new

        return x


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SOLVERS: dict[str, Solver] = {
    # original
    "lsq_linear":    LsqLinearSolver(),
    "nnls":          NNLSSolver(),
    "lasso":         LassoSolver(),
    "ridge":         RidgeSolver(),
    "elastic_net":   ElasticNetSolver(),
    "diffevo":       DiffEvoSolver(),
    # new
    "weighted_ls":     WeightedLSSolver(),
    "pgd":             PGDSolver(),
    "admm":            ADMMSolver(),
    "dual_annealing":  DualAnnealingSolver(),
    "huber":           HuberSolver(),
}

DEFAULT_SOLVER = "lsq_linear"


def get_solver(name: str) -> Solver:
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver '{name}'. Choose from: {', '.join(SOLVERS)}")
    return SOLVERS[name]
