"""Command-line PF-1000 driver for the mlx_mhd package."""

from __future__ import annotations

from mlx_mhd.driver import plot_pf1000, run_pf1000


def main() -> None:
    result = run_pf1000()
    print(f"Peak current: {result.peak_current * 1e-6:.3f} MA")
    plot_pf1000(result)


if __name__ == "__main__":
    main()
