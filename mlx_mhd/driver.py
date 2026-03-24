"""Driver helpers for PF-1000 discharge runs."""

from __future__ import annotations

from typing import Optional

from .solver import PF1000_CIRCUIT, PF1000RunResult, SolverConfig, run_pf1000_simulation


def run_pf1000(
    *,
    total_time: float = 12e-6,
    dt: float = 1e-7,
    nr: int = 64,
    nz: int = 128,
    config: SolverConfig = SolverConfig(),
) -> PF1000RunResult:
    """Convenience wrapper for running the PF-1000 configuration."""

    return run_pf1000_simulation(total_time=total_time, dt=dt, nr=nr, nz=nz, circuit=PF1000_CIRCUIT, config=config)


def plot_pf1000(result: PF1000RunResult, *, output_path: Optional[str] = None) -> None:
    """Plot the discharge current trace when matplotlib is available."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(result.times * 1e6, result.currents * 1e-6, lw=2.0)
    ax.set_xlabel("time [µs]")
    ax.set_ylabel("current [MA]")
    ax.set_title("PF-1000 discharge current")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path, dpi=150)
    plt.close(fig)

