"""Core math and plotting helpers for spectral matching."""

import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go

WAVELENGTHS = np.arange(380, 781, 1)  # 401 points
N_CHANNELS = 24
PER_CHANNEL_DIR = str(pathlib.Path(__file__).parent / "data" / "Per Channel values")

SOLVER_COLORS: dict[str, str] = {
    "lsq_linear":     "#facc15",
    "nnls":           "#4ade80",
    "lasso":          "#f87171",
    "ridge":          "#60a5fa",
    "elastic_net":    "#fb923c",
    "diffevo":        "#e879f9",
    "weighted_ls":    "#e2e8f0",
    "pgd":            "#34d399",
    "admm":           "#f472b6",
    "dual_annealing": "#a78bfa",
    "huber":          "#fbbf24",
}


def _read_channel_file(ch: int) -> tuple[np.ndarray, np.ndarray]:
    path = pathlib.Path(PER_CHANNEL_DIR) / str(ch)
    df = pd.read_csv(path, sep="\t", header=0)
    return df.iloc[:, 0].to_numpy(dtype=float), df.iloc[:, 1].to_numpy(dtype=float)


def _build_channel_wl() -> dict[int, int]:
    result = {}
    for ch in range(1, N_CHANNELS + 1):
        wl, irr = _read_channel_file(ch)
        result[ch] = int(round(wl[np.argmax(irr)]))
    return result


CHANNEL_WL: dict[int, int] = _build_channel_wl()


def load_matrix() -> np.ndarray:
    cols = []
    for ch in range(1, N_CHANNELS + 1):
        wl, irr = _read_channel_file(ch)
        cols.append(np.interp(WAVELENGTHS, wl, irr))
    A = np.column_stack(cols)
    assert A.shape == (401, N_CHANNELS)
    global_max = A.max()
    if global_max > 0:
        A /= global_max
    return A


def load_target(path: str, reference: float | None = None) -> np.ndarray:
    df = pd.read_csv(path, header=None, comment="#")
    wl = df.iloc[:, 0].to_numpy(dtype=float)
    intensity = df.iloc[:, 1].to_numpy(dtype=float)
    t = intensity if np.array_equal(wl, WAVELENGTHS) else np.interp(WAVELENGTHS, wl, intensity)
    divisor = reference if reference is not None else t.max()
    return t / divisor if divisor > 0 else t


def zero_inactive_channels(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Zero coefficients for channels whose column in A is all zeros (no visible emission)."""
    x = x.copy()
    x[A.max(axis=0) == 0] = 0.0
    return x


def seed_from_spectrum(t: np.ndarray) -> np.ndarray:
    wl_min, wl_max = int(WAVELENGTHS[0]), int(WAVELENGTHS[-1])
    x0 = np.array([
        t[CHANNEL_WL[ch] - wl_min] if wl_min <= CHANNEL_WL[ch] <= wl_max else 0.0
        for ch in range(1, 25)
    ])
    peak = x0.max()
    return np.clip(x0 / peak, 0, 1) if peak > 0 else x0


def match_quality(A: np.ndarray, x: np.ndarray, t: np.ndarray) -> tuple[float, float, float]:
    reconstructed = A @ x
    denom = np.linalg.norm(reconstructed) * np.linalg.norm(t)
    cos_a = float(np.clip(np.dot(reconstructed, t) / denom, -1.0, 1.0)) if denom > 0 else 0.0
    sam = float(np.degrees(np.arccos(cos_a)))
    ss_res = float(np.dot(reconstructed - t, reconstructed - t))
    ss_tot = float(np.dot(t - t.mean(), t - t.mean()))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((reconstructed - t) ** 2)))
    return sam, r2, rmse


def wavelength_to_rgb(wl: float) -> str:
    if 380 <= wl < 440:
        r, g, b = -(wl - 440) / 60, 0.0, 1.0
    elif 440 <= wl < 490:
        r, g, b = 0.0, (wl - 440) / 50, 1.0
    elif 490 <= wl < 510:
        r, g, b = 0.0, 1.0, -(wl - 510) / 20
    elif 510 <= wl < 580:
        r, g, b = (wl - 510) / 70, 1.0, 0.0
    elif 580 <= wl < 645:
        r, g, b = 1.0, -(wl - 645) / 65, 0.0
    elif 645 <= wl <= 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        r, g, b = 0.0, 0.0, 0.0

    if 380 <= wl < 420:
        factor = 0.3 + 0.7 * (wl - 380) / 40
    elif 700 < wl <= 780:
        factor = 0.3 + 0.7 * (780 - wl) / 80
    else:
        factor = 1.0

    ri, gi, bi = (int(round(c * factor * 255)) for c in (r, g, b))
    return f"#{ri:02x}{gi:02x}{bi:02x}"


def save_plot(
    wavelengths: np.ndarray,
    target: np.ndarray,
    reconstructed: np.ndarray,
    sam: float,
    r2: float,
    rmse: float,
    out_path: str,
) -> None:
    fig = go.Figure()

    band = 10
    for wl in range(int(wavelengths[0]), int(wavelengths[-1]) + 1, band):
        fig.add_vrect(
            x0=wl, x1=wl + band,
            fillcolor=wavelength_to_rgb(wl + band // 2),
            opacity=0.25,
            layer="below",
            line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=wavelengths, y=target,
        name="Target",
        line=dict(color="white", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=wavelengths, y=reconstructed,
        name=f"Reconstructed (SAM={sam:.1f}°  R²={r2:.3f}  RMSE={rmse:.4f})",
        line=dict(color="yellow", width=2, dash="dash"),
    ))

    fig.update_layout(
        title="Spectral Match",
        xaxis_title="Wavelength (nm)",
        yaxis_title="Intensity",
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font_color="white",
        legend=dict(bgcolor="#222"),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )
    fig.write_html(out_path)


def save_comparison_plot(
    wavelengths: np.ndarray,
    target: np.ndarray,
    solver_results: list[tuple[str, np.ndarray, float, float, float]],
    best_name: str,
    out_path: str,
) -> None:
    """Save a multi-solver plot: best solver highlighted, others as dim dashes."""
    fig = go.Figure()

    band = 10
    for wl in range(int(wavelengths[0]), int(wavelengths[-1]) + 1, band):
        fig.add_vrect(
            x0=wl, x1=wl + band,
            fillcolor=wavelength_to_rgb(wl + band // 2),
            opacity=0.25, layer="below", line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=wavelengths, y=target, name="Target",
        line=dict(color="white", width=2),
    ))

    for name, recon, sam, r2, rmse in solver_results:
        if name == best_name:
            continue
        colour = SOLVER_COLORS.get(name, "#aaaaaa")
        fig.add_trace(go.Scatter(
            x=wavelengths, y=recon,
            name=f"{name}  SAM={sam:.1f}°  R²={r2:.3f}",
            line=dict(color=colour, width=1.5, dash="dash"),
        ))

    for name, recon, sam, r2, rmse in solver_results:
        if name != best_name:
            continue
        colour = SOLVER_COLORS.get(name, "#facc15")
        fig.add_trace(go.Scatter(
            x=wavelengths, y=recon,
            name=f"★ {name}  SAM={sam:.1f}°  R²={r2:.3f}  RMSE={rmse:.4f}",
            line=dict(color=colour, width=4),
        ))

    fig.update_layout(
        title="Solver Comparison",
        xaxis_title="Wavelength (nm)", yaxis_title="Intensity",
        paper_bgcolor="#111", plot_bgcolor="#111", font_color="white",
        legend=dict(bgcolor="#222", font=dict(size=11)),
        xaxis=dict(gridcolor="#333"), yaxis=dict(gridcolor="#333"),
    )
    fig.write_html(out_path)
