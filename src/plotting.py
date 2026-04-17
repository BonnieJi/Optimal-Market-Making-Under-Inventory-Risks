"""Figures for one path and Monte Carlo distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_sample_path(state, title: str, save_path: Path | None = None) -> None:
    t = np.asarray(state.hist_t, dtype=float)
    S = np.asarray(state.hist_S, dtype=float)
    q = np.asarray(state.hist_q, dtype=float)
    W = np.asarray(state.hist_W, dtype=float)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    axes[0].plot(t, S, color="C0", lw=1.0)
    axes[0].set_ylabel("Mid S")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, q, color="C1", lw=1.0)
    axes[1].set_ylabel("Inventory q")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, W, color="C2", lw=1.0)
    axes[2].set_ylabel("Wealth W")
    axes[2].set_xlabel("t")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_terminal_pnl_histogram(
    pnl_baseline: np.ndarray,
    pnl_heuristic: np.ndarray,
    *,
    labels: tuple[str, str] = ("baseline", "inventory-aware"),
    title: str = "Terminal PnL (W_T - W_0)",
    save_path: Path | None = None,
    bins: int = 40,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        pnl_baseline,
        bins=bins,
        alpha=0.55,
        label=labels[0],
        color="C0",
        density=True,
    )
    ax.hist(
        pnl_heuristic,
        bins=bins,
        alpha=0.55,
        label=labels[1],
        color="C1",
        density=True,
    )
    ax.set_xlabel("Terminal PnL")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_drawdown_histogram(
    dd_baseline: np.ndarray,
    dd_heuristic: np.ndarray,
    *,
    labels: tuple[str, str] = ("baseline", "inventory-aware"),
    title: str = "Max drawdown per path (on PnL)",
    save_path: Path | None = None,
    bins: int = 40,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        dd_baseline,
        bins=bins,
        alpha=0.55,
        label=labels[0],
        color="C0",
        density=True,
    )
    ax.hist(
        dd_heuristic,
        bins=bins,
        alpha=0.55,
        label=labels[1],
        color="C1",
        density=True,
    )
    ax.set_xlabel("Max drawdown (<= 0)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sensitivity_twinx(
    x: np.ndarray,
    y_left: np.ndarray,
    y_right: np.ndarray,
    *,
    xlabel: str,
    left_ylabel: str,
    right_ylabel: str,
    title: str,
    save_path: Path | None = None,
) -> None:
    """Compare two metrics vs one swept parameter (e.g. terminal PnL vs mean half-spread)."""
    fig, ax_l = plt.subplots(figsize=(8, 5))
    ax_l.plot(x, y_left, color="C0", marker="o", ms=4, lw=1.5)
    ax_l.set_xlabel(xlabel)
    ax_l.set_ylabel(left_ylabel, color="C0")
    ax_l.tick_params(axis="y", labelcolor="C0")
    ax_l.grid(True, alpha=0.3)

    ax_r = ax_l.twinx()
    ax_r.plot(x, y_right, color="C1", marker="s", ms=4, lw=1.5)
    ax_r.set_ylabel(right_ylabel, color="C1")
    ax_r.tick_params(axis="y", labelcolor="C1")

    ax_l.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sensitivity_line(
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, color="C0", marker="o", ms=4, lw=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)
