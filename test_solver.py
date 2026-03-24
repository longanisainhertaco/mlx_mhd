from __future__ import annotations

import math

import numpy as np

import mlx_mhd as mhd


def _grid() -> mhd.CylindricalGrid:
    return mhd.CylindricalGrid(nr=12, nz=16, dr=0.01, dz=0.015)


def _primitive_state(grid: mhd.CylindricalGrid) -> np.ndarray:
    r = grid.radii[:, None]
    z = grid.axial[None, :]
    prim = mhd.make_uniform_primitive(grid, rho=1.0, pressure=1.0, bz=0.2, electron_fraction=0.2)
    prim[mhd.COMPONENTS["vr"]] = 0.05 * np.sin(np.pi * r)
    prim[mhd.COMPONENTS["vz"]] = 0.03 * np.cos(np.pi * z / max(grid.axial[-1], grid.dz))
    prim[mhd.COMPONENTS["vtheta"]] = 0.02 * r
    prim[mhd.COMPONENTS["Btheta"]] = 0.01 / np.maximum(r, 1e-3)
    return prim.astype(np.float32)


def test_dual_energy_pressure_recovery_stays_positive_in_low_beta_limit():
    grid = _grid()
    prim = mhd.make_uniform_primitive(grid, rho=1.0, pressure=1e-6, bz=24.0, electron_fraction=0.1)
    state = mhd.primitive_to_conserved(prim)
    pressure = mhd.recover_pressure(state)
    assert np.all(np.isfinite(pressure))
    assert np.min(pressure) > 0.0


def test_conserved_primitive_round_trip_is_stable():
    grid = _grid()
    prim = _primitive_state(grid)
    state = mhd.primitive_to_conserved(prim)
    recovered = mhd.conserved_to_primitive(state)
    np.testing.assert_allclose(recovered[0], prim[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(recovered[1:4], prim[1:4], rtol=5e-4, atol=5e-5)
    np.testing.assert_allclose(recovered[4], prim[4], rtol=5e-4, atol=5e-5)


def test_weno5z_preserves_constant_stencil():
    stencil = np.ones((10, 5, 7), dtype=np.float32)
    recon = mhd.weno5z_left(stencil)
    np.testing.assert_allclose(recon, 1.0)


def test_flux_divergence_is_finite_for_smooth_state():
    grid = _grid()
    state = mhd.primitive_to_conserved(_primitive_state(grid))
    flux_r, flux_z, div = mhd.compute_flux_divergence(state, grid, current_I=10_000.0)
    assert flux_r.shape == (10, grid.nr + 1, grid.nz)
    assert flux_z.shape == (10, grid.nr, grid.nz + 1)
    assert div.shape == state.shape
    assert np.all(np.isfinite(div))


def test_entropy_resync_raises_shocked_entropy_when_energy_pressure_is_large():
    grid = _grid()
    base = mhd.primitive_to_conserved(_primitive_state(grid))
    shocked = base.copy()
    shocked[mhd.COMPONENTS["rho_vr"], 5, 5] = -4.0
    shocked[mhd.COMPONENTS["E"], 5, 5] = 20.0
    shocked[mhd.COMPONENTS["Srho"], 5, 5] *= 0.1
    synced = mhd.entropy_resynchronize(base, shocked, grid)
    assert synced[mhd.COMPONENTS["Srho"], 5, 5] >= shocked[mhd.COMPONENTS["Srho"], 5, 5]


def test_constrained_transport_preserves_uniform_divergence_free_field():
    grid = _grid()
    prim = mhd.make_uniform_primitive(grid, rho=1.0, pressure=1.0, br=0.0, bz=0.1)
    state = mhd.primitive_to_conserved(prim)
    updated = mhd.constrained_transport_update(state, grid, dt=1e-8)
    divb = mhd.divergence_b(updated, grid)
    assert np.max(np.abs(divb)) < 1e-5


def test_resistive_diffusion_smooths_magnetic_gradients():
    grid = _grid()
    prim = _primitive_state(grid)
    prim[mhd.COMPONENTS["Br"]] = np.linspace(0.0, 1.0, grid.nr, dtype=np.float32)[:, None]
    state = mhd.primitive_to_conserved(prim)
    cfg = mhd.SolverConfig(resistivity=1e-3)
    diffused = mhd.implicit_resistive_diffusion(state, grid, dt=1e-3, config=cfg)
    assert np.var(diffused[mhd.COMPONENTS["Br"]]) < np.var(state[mhd.COMPONENTS["Br"]])


def test_plasma_inductance_is_monotonic_when_previous_value_is_supplied():
    grid = _grid()
    state = mhd.initialize_pf1000_state(grid)
    lp = mhd.extract_plasma_inductance(state, grid, mhd.PF1000_CIRCUIT)
    lp_clamped = mhd.extract_plasma_inductance(state, grid, mhd.PF1000_CIRCUIT, previous_inductance=lp + 1e-8)
    assert lp_clamped >= lp + 1e-8


def test_circuit_step_reports_back_emf_from_inductance_growth():
    circuit = mhd.CircuitState(current=1e5, capacitor_voltage=mhd.PF1000_CIRCUIT.V0, plasma_inductance=10e-9)
    updated = mhd.step_circuit(circuit, mhd.PF1000_CIRCUIT, dt=1e-8, plasma_inductance=12e-9)
    expected = -circuit.current * (12e-9 - 10e-9) / 1e-8
    assert math.isclose(updated.back_emf, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_solver_step_keeps_state_finite_and_pressure_positive():
    grid = _grid()
    cfg = mhd.SolverConfig(resistivity=1e-5)
    solver = mhd.MHDSolver(grid, config=cfg)
    prim = mhd.make_uniform_primitive(grid, rho=1.0, pressure=1e-4, bz=5.0, btheta=0.5 / grid.radii[:, None])
    state = mhd.primitive_to_conserved(prim)
    dt = min(1e-8, solver.courant_timestep(state))
    new_state, _ = solver.step(state, dt)
    pressure = mhd.recover_pressure(new_state)
    assert np.all(np.isfinite(new_state))
    assert np.min(pressure) > 0.0


def test_pf1000_driver_smoke_run_returns_reasonable_current_trace():
    result = mhd.run_pf1000(total_time=2e-7, dt=5e-8, nr=12, nz=24)
    assert result.times.shape == result.currents.shape
    assert np.all(np.isfinite(result.currents))
    assert result.peak_current >= 0.0
