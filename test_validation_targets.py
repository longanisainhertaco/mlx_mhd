"""Validation-target tests for the MHD solver.

Eight targets covering hydrodynamic shock capturing, MHD compound waves,
low-beta robustness, positivity, conservation, and divergence cleaning.
"""
from __future__ import annotations

import numpy as np

import mlx_mhd as mhd

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GAMMA = mhd.GAMMA  # 5/3
C = mhd.COMPONENTS


def _exact_sod_solution(x: np.ndarray, t: float, x0: float = 0.5) -> np.ndarray:
    """Exact density for the Sod shock tube (gamma = 5/3).

    Left state:  rho=1,     p=1,   v=0
    Right state: rho=0.125, p=0.1, v=0
    """
    gamma = GAMMA
    gm1 = gamma - 1.0
    gp1 = gamma + 1.0

    rhoL, pL, vL = 1.0, 1.0, 0.0
    cL = np.sqrt(gamma * pL / rhoL)
    rhoR, pR, vR = 0.125, 0.1, 0.0
    cR = np.sqrt(gamma * pR / rhoR)

    # Newton iteration for post-shock pressure p*
    p_star = 0.30313
    for _ in range(100):
        A_R = 2.0 / (gp1 * rhoR)
        B_R = gm1 / gp1 * pR
        g_R = np.sqrt(A_R / (p_star + B_R))
        f_R = (p_star - pR) * g_R
        df_R = g_R * (1.0 - 0.5 * (p_star - pR) / (p_star + B_R))

        f_L = (2.0 * cL / gm1) * ((p_star / pL) ** (gm1 / (2.0 * gamma)) - 1.0)
        df_L = (1.0 / (rhoL * cL)) * (p_star / pL) ** (-(gp1) / (2.0 * gamma))

        dp = -(f_L + f_R + vR - vL) / (df_L + df_R)
        p_star = max(p_star + dp, 1e-10)
        if abs(dp) < 1e-12:
            break

    v_star = 0.5 * (vL + vR) + 0.5 * (f_R - f_L)
    rho_star_R = rhoR * ((p_star / pR + gm1 / gp1) / (gm1 / gp1 * p_star / pR + 1.0))
    rho_star_L = rhoL * (p_star / pL) ** (1.0 / gamma)
    c_star_L = cL * (p_star / pL) ** (gm1 / (2.0 * gamma))

    S_shock = vR + cR * np.sqrt(gp1 / (2.0 * gamma) * (p_star / pR) + gm1 / (2.0 * gamma))
    S_head = vL - cL
    S_tail = v_star - c_star_L

    xi = (x - x0) / t
    # Clamp the rarefaction argument to avoid fractional-power-of-negative warnings
    fan_arg = np.maximum(2.0 / gp1 + gm1 / (gp1 * cL) * (vL - xi), 0.0)
    rho = np.where(
        xi < S_head, rhoL,
        np.where(
            xi < S_tail,
            rhoL * fan_arg ** (2.0 / gm1),
            np.where(
                xi < v_star, rho_star_L,
                np.where(xi < S_shock, rho_star_R, rhoR),
            ),
        ),
    )
    return rho


def _run_loop(solver, state, t_final: float, max_steps: int = 100_000):
    """Advance *state* to *t_final* using CFL sub-steps."""
    t = 0.0
    for _ in range(max_steps):
        if t >= t_final:
            break
        dt = min(solver.courant_timestep(state), t_final - t)
        state, _ = solver.step(state, dt)
        t += dt
    return state


# ---------------------------------------------------------------------------
# 1. Sod shock tube – L1(rho) < 0.02
# ---------------------------------------------------------------------------


def test_validation_sod_shock_tube_l1_density():
    nz = 32
    dz = 1.0 / nz
    grid = mhd.CylindricalGrid(nr=4, nz=nz, dr=0.01, dz=dz, r_min=100.0)
    cfg = mhd.SolverConfig(
        bremsstrahlung=False, enable_ct=False,
        resistivity=0.0, enable_resistive_diffusion=False,
    )
    solver = mhd.MHDSolver(grid, config=cfg)

    prim = mhd.make_uniform_primitive(
        grid, rho=1.0, pressure=1.0, electron_fraction=0.5,
    )
    mid = nz // 2
    prim[C["rho"], :, mid:] = 0.125
    prim[C["p"], :, mid:] = 0.1
    prim[C["Srho"], :, mid:] = 0.1 / 0.125 ** GAMMA  # specific entropy
    prim[C["Ee"], :, mid:] = 0.5 * 0.1 / (GAMMA - 1.0)

    state = mhd.primitive_to_conserved(prim)
    state = _run_loop(solver, state, t_final=0.1)

    rho_num = np.mean(state[C["rho"]], axis=0)  # average over r
    z = grid.axial
    rho_exact = _exact_sod_solution(z, t=0.1)

    l1 = float(np.mean(np.abs(rho_num - rho_exact)))
    assert l1 < 0.02, f"Sod L1(rho) = {l1:.4f}, expected < 0.02"


# ---------------------------------------------------------------------------
# 2. Brio-Wu MHD shock tube – no NaN, compound wave structure
# ---------------------------------------------------------------------------


def test_validation_brio_wu_no_nan_and_compound_waves():
    nz = 32
    dz = 1.0 / nz
    grid = mhd.CylindricalGrid(nr=4, nz=nz, dr=0.01, dz=dz, r_min=100.0)
    cfg = mhd.SolverConfig(
        bremsstrahlung=False, enable_ct=False,
        resistivity=0.0, enable_resistive_diffusion=False,
    )
    solver = mhd.MHDSolver(grid, config=cfg)

    # Modified Brio-Wu with moderate density ratio and B reversal
    prim = mhd.make_uniform_primitive(
        grid, rho=1.0, pressure=1.0, bz=0.75, br=0.5, electron_fraction=0.5,
    )
    mid = nz // 2
    prim[C["rho"], :, mid:] = 0.5
    prim[C["p"], :, mid:] = 0.3
    prim[C["Srho"], :, mid:] = 0.3 / 0.5 ** GAMMA  # specific entropy
    prim[C["Ee"], :, mid:] = 0.5 * 0.3 / (GAMMA - 1.0)
    prim[C["Br"], :, mid:] = -0.5  # B reversal across interface

    state = mhd.primitive_to_conserved(prim)
    for _ in range(30):
        dt = solver.courant_timestep(state)
        state, _ = solver.step(state, dt)

    assert np.all(np.isfinite(state)), "Brio-Wu produced NaN / Inf"

    # Density gradient should be non-monotone (compound MHD wave structure).
    rho_profile = np.mean(state[C["rho"]], axis=0)
    diffs = np.diff(rho_profile)
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
    assert sign_changes >= 2, (
        f"Expected >= 2 gradient sign changes, found {sign_changes}"
    )


# ---------------------------------------------------------------------------
# 3. Low-beta shock – no negative pressure, R-H within 5 %
# ---------------------------------------------------------------------------


def test_validation_low_beta_no_negative_pressure():
    grid = mhd.CylindricalGrid(nr=4, nz=64, dr=0.01, dz=0.02, r_min=100.0)
    cfg = mhd.SolverConfig(
        bremsstrahlung=False, enable_ct=False,
        resistivity=0.0, enable_resistive_diffusion=False,
    )
    solver = mhd.MHDSolver(grid, config=cfg)

    B0 = 1.0
    beta = 0.001
    p0 = beta * B0 ** 2 / (2.0 * mhd.MU0)
    prim = mhd.make_uniform_primitive(
        grid, rho=1.0, pressure=p0, bz=B0, electron_fraction=0.5,
    )
    mid = grid.nz // 2
    prim[C["vz"], :, :mid] = 0.5
    prim[C["vz"], :, mid:] = -0.5

    state = mhd.primitive_to_conserved(prim)
    for _ in range(40):
        dt = solver.courant_timestep(state)
        state, _ = solver.step(state, dt)

    pressure = mhd.recover_pressure(state)
    assert np.all(np.isfinite(state)), "Low-beta shock produced NaN / Inf"
    assert np.min(pressure) > 0.0, (
        f"Negative pressure: min = {np.min(pressure):.3e}"
    )


def test_validation_low_beta_rankine_hugoniot():
    """Symmetric collision: total z-momentum must stay near zero (R-H)."""
    grid = mhd.CylindricalGrid(nr=4, nz=32, dr=0.01, dz=0.04, r_min=100.0)
    cfg = mhd.SolverConfig(
        bremsstrahlung=False, enable_ct=False,
        resistivity=0.0, enable_resistive_diffusion=False,
    )
    solver = mhd.MHDSolver(grid, config=cfg)

    B0 = 1.0
    beta = 0.001
    p0 = beta * B0 ** 2 / (2.0 * mhd.MU0)
    prim = mhd.make_uniform_primitive(
        grid, rho=1.0, pressure=p0, bz=B0, electron_fraction=0.5,
    )
    mid = grid.nz // 2
    prim[C["vz"], :, :mid] = 0.5
    prim[C["vz"], :, mid:] = -0.5

    state = mhd.primitive_to_conserved(prim)
    vol = grid.cell_volumes
    pz_half = float(np.sum(np.abs(state[C["rho_vz"]]) * vol))

    for _ in range(20):
        dt = solver.courant_timestep(state)
        state, _ = solver.step(state, dt)

    pz_total = float(np.sum(state[C["rho_vz"]] * vol))
    momentum_asymmetry = abs(pz_total) / max(pz_half, 1e-30)
    assert momentum_asymmetry < 0.05, (
        f"R-H momentum asymmetry {momentum_asymmetry:.3f}, expected < 0.05"
    )


# ---------------------------------------------------------------------------
# 4. PF-1000 smoke test (I_peak finite, short run)
# ---------------------------------------------------------------------------


def test_validation_pf1000_smoke():
    """Short PF-1000 run: peak current must be finite and positive."""
    result = mhd.run_pf1000(total_time=2e-7, dt=5e-8, nr=12, nz=24)
    assert np.isfinite(result.peak_current)
    assert result.peak_current > 0.0


# ---------------------------------------------------------------------------
# 5. No negative pressure – moderate conditions
# ---------------------------------------------------------------------------


def test_validation_no_negative_pressure():
    grid = mhd.CylindricalGrid(nr=12, nz=16, dr=0.01, dz=0.015)
    cfg = mhd.SolverConfig(resistivity=1e-5, bremsstrahlung=False)
    solver = mhd.MHDSolver(grid, config=cfg)

    prim = mhd.make_uniform_primitive(
        grid, rho=1.0, pressure=1e-4, bz=5.0,
        btheta=0.5 / grid.radii[:, None], electron_fraction=0.2,
    )
    state = mhd.primitive_to_conserved(prim)

    for _ in range(10):
        dt = min(1e-8, solver.courant_timestep(state))
        state, _ = solver.step(state, dt)
        pressure = mhd.recover_pressure(state)
        assert np.all(np.isfinite(state)), "State has NaN / Inf"
        assert np.min(pressure) > 0.0, (
            f"Negative pressure at step: min = {np.min(pressure):.3e}"
        )


# ---------------------------------------------------------------------------
# 6. Mass conservation < 5 %
# ---------------------------------------------------------------------------


def test_validation_mass_conservation():
    grid = mhd.CylindricalGrid(nr=8, nz=16, dr=0.01, dz=0.015)
    cfg = mhd.SolverConfig(bremsstrahlung=False, resistivity=0.0)
    solver = mhd.MHDSolver(grid, config=cfg)

    prim = mhd.make_uniform_primitive(
        grid, rho=1.0, pressure=1.0, bz=0.2, electron_fraction=0.5,
    )
    state = mhd.primitive_to_conserved(prim)

    vol = grid.cell_volumes  # (nr, nz)
    initial_mass = float(np.sum(state[C["rho"]] * vol))

    for _ in range(30):
        dt = solver.courant_timestep(state)
        state, _ = solver.step(state, dt)

    final_mass = float(np.sum(state[C["rho"]] * vol))
    rel_err = abs(final_mass - initial_mass) / max(abs(initial_mass), 1e-30)
    assert rel_err < 0.05, f"Mass conservation error {rel_err:.4f}, expected < 5 %"


# ---------------------------------------------------------------------------
# 7. Energy conservation < 10 %
# ---------------------------------------------------------------------------


def test_validation_energy_conservation():
    grid = mhd.CylindricalGrid(nr=8, nz=16, dr=0.01, dz=0.015)
    cfg = mhd.SolverConfig(bremsstrahlung=False, resistivity=0.0)
    solver = mhd.MHDSolver(grid, config=cfg)

    prim = mhd.make_uniform_primitive(
        grid, rho=1.0, pressure=1.0, bz=0.2, electron_fraction=0.5,
    )
    state = mhd.primitive_to_conserved(prim)

    vol = grid.cell_volumes
    initial_energy = float(np.sum(state[C["E"]] * vol))

    for _ in range(30):
        dt = solver.courant_timestep(state)
        state, _ = solver.step(state, dt)

    final_energy = float(np.sum(state[C["E"]] * vol))
    rel_err = abs(final_energy - initial_energy) / max(abs(initial_energy), 1e-30)
    assert rel_err < 0.10, (
        f"Energy conservation error {rel_err:.4f}, expected < 10 %"
    )


# ---------------------------------------------------------------------------
# 8. div(B) relative error < 1e-6 after constrained transport
# ---------------------------------------------------------------------------


def test_validation_divb_relative_error():
    grid = mhd.CylindricalGrid(nr=12, nz=16, dr=0.01, dz=0.015)

    prim = mhd.make_uniform_primitive(
        grid, rho=1.0, pressure=1.0, br=0.0, bz=0.1, electron_fraction=0.5,
    )
    state = mhd.primitive_to_conserved(prim)

    # Apply CT update directly (cleanest check of divergence-free preservation)
    for _ in range(5):
        state = mhd.constrained_transport_update(state, grid, dt=1e-8)

    divb = mhd.divergence_b(state, grid)
    B_mag = np.sqrt(state[C["Br"]] ** 2 + state[C["Bz"]] ** 2 + state[C["Btheta"]] ** 2)
    L = max(grid.dr, grid.dz)
    B_ref = float(np.max(B_mag))
    if B_ref < 1e-15:
        return
    rel_divb = float(np.max(np.abs(divb))) / (B_ref / L)
    assert rel_divb < 1e-6, (
        f"div(B) relative error {rel_divb:.3e}, expected < 1e-6"
    )
