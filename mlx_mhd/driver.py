"""Driver helpers for PF-1000 discharge runs."""

from __future__ import annotations

from typing import Optional

from .solver import (
    PF1000_CIRCUIT,
    PF1000RunResult,
    SolverConfig,
    run_pf1000_simulation,
    validate_pf1000,
)


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


def print_validation_report(result: PF1000RunResult) -> None:
    """Print a human-readable validation report for all 8 targets."""
    report = validate_pf1000(result)
    print("PF-1000 Validation Report")
    print("=" * 60)
    for name, (value, passes) in report.items():
        status = "PASS" if passes else "FAIL"
        print(f"  [{status}] {name}: {value:.4g}")
    n_pass = sum(1 for _, p in report.values() if p)
    print("-" * 60)
    print(f"  {n_pass}/{len(report)} targets within reference range")

