from __future__ import annotations

import numpy as np

import mlx_mhd as mhd
from mhd_reference import GAMMA, ghost_pad_np, hlld_flux_np, cyl_source_np


def _make_conserved_state(nr: int = 8, nz: int = 4, dr: float = 0.01) -> np.ndarray:
    r = (np.arange(nr) + 0.5) * dr
    r2d = r[:, None]
    ones = np.ones((nr, nz), dtype=np.float32)

    rho = (1.0 + 0.1 * np.sin(np.pi * r2d)).astype(np.float32) * ones
    vr = (0.05 * np.cos(np.pi * r2d)).astype(np.float32) * ones
    vz = np.float32(0.02) * ones
    vtheta = (0.1 * r2d).astype(np.float32) * ones
    Br = np.float32(0.01) * ones
    Bz = np.float32(0.5) * ones
    Btheta = (0.02 / (r2d + 1e-10)).astype(np.float32) * ones
    p = (1.0 + 0.05 * np.sin(2.0 * np.pi * r2d)).astype(np.float32) * ones

    B2 = Br**2 + Bz**2 + Btheta**2
    v2 = vr**2 + vz**2 + vtheta**2
    E = p / np.float32(GAMMA - 1.0) + np.float32(0.5) * rho * v2 + np.float32(0.5) * B2
    Srho = rho * p ** np.float32(1.0 / GAMMA)
    Ee = np.float32(0.3) * p / np.float32(GAMMA - 1.0)

    return np.stack(
        [rho, rho * vr, rho * vz, rho * vtheta, E, Srho, Br, Bz, Btheta, Ee],
        axis=0,
    ).astype(np.float32)


def _make_primitive_state(nr: int = 8, nz: int = 4, dr: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    r = (np.arange(nr) + 0.5) * dr
    r2d = r[:, None]
    ones = np.ones((nr, nz), dtype=np.float32)

    rho = (1.0 + 0.1 * np.sin(np.pi * r2d)).astype(np.float32) * ones
    vr = (0.05 * np.cos(np.pi * r2d)).astype(np.float32) * ones
    vz = np.float32(0.02) * ones
    vtheta = (0.1 * r2d).astype(np.float32) * ones
    p = (1.0 + 0.05 * np.sin(2.0 * np.pi * r2d)).astype(np.float32) * ones
    S = p / rho ** np.float32(GAMMA)
    Br = np.float32(0.01) * ones
    Bz = np.float32(0.5) * ones
    Btheta = (0.02 / (r2d + 1e-10)).astype(np.float32) * ones
    Ee = np.float32(0.3) * p / np.float32(GAMMA - 1.0)

    prim = np.stack([rho, vr, vz, vtheta, p, S, Br, Bz, Btheta, Ee], axis=0).astype(np.float32)
    return prim, r.astype(np.float32)


def test_components_expose_conserved_layout_and_compatibility_aliases():
    assert mhd.COMPONENTS["rho"] == 0
    assert mhd.COMPONENTS["rho_vr"] == 1
    assert mhd.COMPONENTS["rho_vz"] == 2
    assert mhd.COMPONENTS["rho_vtheta"] == 3
    assert mhd.COMPONENTS["E"] == 4
    assert mhd.COMPONENTS["Srho"] == 5
    assert mhd.COMPONENTS["Br"] == 6
    assert mhd.COMPONENTS["Bz"] == 7
    assert mhd.COMPONENTS["Btheta"] == 8
    assert mhd.COMPONENTS["Ee"] == 9

    # Backward-compatible aliases used by primitive-variable helpers.
    assert mhd.COMPONENTS["vr"] == mhd.COMPONENTS["rho_vr"]
    assert mhd.COMPONENTS["vz"] == mhd.COMPONENTS["rho_vz"]
    assert mhd.COMPONENTS["vtheta"] == mhd.COMPONENTS["rho_vtheta"]
    assert mhd.COMPONENTS["p"] == mhd.COMPONENTS["E"]


def test_package_ghost_pad_numpy_matches_reference():
    state = _make_conserved_state()
    expected = ghost_pad_np(state, I=1e4, dr=0.01, ng=3)
    actual = mhd.ghost_pad_numpy(state, current_I=1e4, dr=0.01, ng=3)
    np.testing.assert_allclose(actual, expected)


def test_package_hlld_flux_numpy_supports_direction_and_matches_reference():
    state = _make_conserved_state()
    left = state
    right_r = np.roll(state, -1, axis=1).astype(np.float32)
    right_z = np.roll(state, -1, axis=2).astype(np.float32)

    np.testing.assert_allclose(
        mhd.hlld_flux_numpy(left, right_r, direction=0),
        hlld_flux_np(left, right_r, direction=0),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        mhd.hlld_flux_numpy(left, right_z, direction=1),
        hlld_flux_np(left, right_z, direction=1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_package_cylindrical_sources_numpy_infers_dr_and_matches_reference():
    prim, radii = _make_primitive_state()
    expected = cyl_source_np(prim, radii, dr=float(radii[1] - radii[0]))
    actual = mhd.cylindrical_sources_numpy(prim, radii)
    np.testing.assert_allclose(actual, expected)
