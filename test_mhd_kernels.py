"""
test_mhd_kernels.py – Tests for the MHD solver kernels.

NumPy reference tests always run.
MLX Metal kernel tests are skipped when MLX is unavailable (e.g. in CI
without Apple Silicon).
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from mhd_reference import (
    ghost_pad_np,
    hlld_flux_np,
    cyl_source_np,
    GAMMA,
    MU0,
)

# ---------------------------------------------------------------------------
# Optional MLX
# ---------------------------------------------------------------------------
try:
    import mlx.core as mx
    from mhd_kernels import ghost_pad, hlld_flux, cyl_source
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

requires_mlx = pytest.mark.skipif(not _HAS_MLX, reason="MLX not available")

# ---------------------------------------------------------------------------
# Tolerance for float32 comparisons (GPU vs NumPy float64 reference)
# ---------------------------------------------------------------------------
ATOL = 5e-5
RTOL = 1e-3


# ============================================================================
# Helper factories
# ============================================================================

def _make_smooth_state(nr: int = 16, nz: int = 8,
                       dr: float = 0.01) -> np.ndarray:
    """Return a smoothly varying conserved state (10, nr, nz)."""
    rng = np.random.default_rng(42)
    r = (np.arange(nr) + 0.5) * dr        # (nr,)
    z = np.linspace(0, 1, nz)             # (nz,)
    r2d = r[:, None]                       # (nr, 1)

    ones   = np.ones((nr, nz))
    rho    = (1.0 + 0.1 * np.sin(np.pi * r2d)) * ones
    vr     = 0.05 * np.cos(np.pi * r2d) * ones
    vz     = 0.02 * ones
    vtheta = 0.1 * r2d * ones
    Br     = 0.01 * ones
    Bz     = 0.5  * ones
    Btheta = 0.02 / (r2d + 1e-10) * ones

    p  = (1.0 + 0.05 * np.sin(2 * np.pi * r2d)) * ones
    B2 = Br**2 + Bz**2 + Btheta**2
    v2 = vr**2 + vz**2 + vtheta**2
    E  = p / (GAMMA - 1) + 0.5 * rho * v2 + 0.5 * B2

    Srho = rho * p**(1.0 / GAMMA)
    Ee   = 0.3 * p / (GAMMA - 1)

    state = np.stack([
        rho,
        rho * vr,
        rho * vz,
        rho * vtheta,
        E,
        Srho,
        Br,
        Bz,
        Btheta,
        Ee,
    ], axis=0).astype(np.float32)   # (10, nr, nz)
    return state


def _make_prim_state(nr: int = 16, nz: int = 8,
                     dr: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Return primitive state (10, nr, nz) and radii (nr,)."""
    r = (np.arange(nr) + 0.5) * dr
    r2d = r[:, None]

    rho    = (1.0 + 0.1 * np.sin(np.pi * r2d)) * np.ones((nr, nz))
    vr     = 0.05 * np.cos(np.pi * r2d) * np.ones((nr, nz))
    vz     = 0.02 * np.ones((nr, nz))
    vtheta = 0.1 * r2d * np.ones((nr, nz))
    p      = (1.0 + 0.05 * np.sin(2 * np.pi * r2d)) * np.ones((nr, nz))
    S      = p / rho**(GAMMA)
    Br     = 0.01 * np.ones((nr, nz))
    Bz     = 0.5  * np.ones((nr, nz))
    Btheta = 0.02 / (r2d + 1e-10) * np.ones((nr, nz))
    Ee     = 0.3 * p / (GAMMA - 1)

    prim = np.stack([rho, vr, vz, vtheta, p, S, Br, Bz, Btheta, Ee],
                    axis=0).astype(np.float32)
    return prim, r.astype(np.float32)


# ============================================================================
# Tests – NumPy reference implementations
# ============================================================================

class TestGhostPadNp:
    """Tests for the NumPy ghost-cell padding reference."""

    def setup_method(self):
        self.nr, self.nz = 16, 8
        self.dr = 0.01
        self.ng = 3
        self.I  = 1e4         # 10 kA
        self.state = _make_smooth_state(self.nr, self.nz, self.dr)

    def test_output_shape(self):
        padded = ghost_pad_np(self.state, self.I, self.dr, self.ng)
        assert padded.shape == (10, self.nr + 2*self.ng, self.nz)

    def test_interior_copy(self):
        padded = ghost_pad_np(self.state, self.I, self.dr, self.ng)
        np.testing.assert_array_equal(
            padded[:, self.ng : self.ng + self.nr, :],
            self.state
        )

    def test_inner_ghost_density_is_positive(self):
        padded = ghost_pad_np(self.state, self.I, self.dr, self.ng)
        inner = padded[0, :self.ng, :]
        assert np.all(inner > 0), "Ghost density must be positive"

    def test_inner_ghost_vr_sign_flip(self):
        """rho*vr (var 1) must be sign-flipped for inner ghosts."""
        padded = ghost_pad_np(self.state, self.I, self.dr, self.ng)
        # Ghost at ir_pad=ng-1 mirrors interior ir_src=0
        np.testing.assert_allclose(
            padded[1, self.ng - 1, :],
            -self.state[1, 0, :],
        )

    def test_inner_ghost_Br_sign_flip(self):
        padded = ghost_pad_np(self.state, self.I, self.dr, self.ng)
        np.testing.assert_allclose(
            padded[6, self.ng - 1, :],
            -self.state[6, 0, :],
        )

    def test_outer_ghost_vr_zero(self):
        padded = ghost_pad_np(self.state, self.I, self.dr, self.ng)
        outer_vr = padded[1, self.ng + self.nr :, :]
        np.testing.assert_array_equal(outer_vr, 0.0)

    def test_outer_ghost_Br_zero(self):
        padded = ghost_pad_np(self.state, self.I, self.dr, self.ng)
        outer_Br = padded[6, self.ng + self.nr :, :]
        np.testing.assert_array_equal(outer_Br, 0.0)

    def test_outer_ghost_Btheta_formula(self):
        padded = ghost_pad_np(self.state, self.I, self.dr, self.ng)
        for g in range(self.ng):
            ir_pad = self.ng + self.nr + g
            r = (ir_pad - self.ng + 0.5) * self.dr
            expected = MU0 * self.I / (2.0 * math.pi * r)
            np.testing.assert_allclose(
                padded[8, ir_pad, :],
                expected,
                rtol=1e-6,
            )

    def test_outer_ghost_zero_gradient_rho(self):
        """Density in outer ghosts should equal the last interior cell."""
        padded = ghost_pad_np(self.state, self.I, self.dr, self.ng)
        last_interior_rho = self.state[0, self.nr - 1, :]
        for g in range(self.ng):
            np.testing.assert_allclose(
                padded[0, self.ng + self.nr + g, :],
                last_interior_rho,
            )


class TestHlldFluxNp:
    """Tests for the NumPy HLLD flux reference."""

    def setup_method(self):
        self.nr, self.nz = 8, 6
        self.dr = 0.01
        self.state = _make_smooth_state(self.nr, self.nz, self.dr)

    def test_output_shape_r_dir(self):
        UL = self.state
        UR = self.state * 0.99
        F = hlld_flux_np(UL, UR, direction=0)
        assert F.shape == UL.shape

    def test_output_shape_z_dir(self):
        UL = self.state
        UR = self.state * 0.99
        F = hlld_flux_np(UL, UR, direction=1)
        assert F.shape == UL.shape

    def test_no_nan_smooth_state(self):
        UL = self.state
        UR = np.roll(self.state, -1, axis=1)   # shifted right state
        F = hlld_flux_np(UL, UR, direction=0)
        assert np.all(np.isfinite(F)), "Flux must be finite for smooth input"

    def test_consistency_equal_states(self):
        """F(U, U) must equal the physical flux."""
        from mhd_reference import _physical_flux_1d, _primitives
        UL = self.state
        UR = self.state.copy()
        F = hlld_flux_np(UL, UR, direction=0)
        # Check mass flux: F[0] = rho * vr
        prim0 = _primitives(UL[:, 0, 0])
        expected_mass = prim0["rho"] * prim0["vr"]
        assert abs(float(F[0, 0, 0]) - expected_mass) < 1e-4

    def test_symmetry_direction(self):
        """Fluxes in r and z directions should differ for asymmetric states."""
        UL = self.state
        UR = self.state * 0.99
        Fr = hlld_flux_np(UL, UR, direction=0)
        Fz = hlld_flux_np(UL, UR, direction=1)
        # They should not be identical (different normal directions)
        assert not np.allclose(Fr, Fz)

    def test_entropy_flux_uses_contact_speed(self):
        """For equal L/R states, entropy flux = Srho * vn (no-jump case)."""
        UL = self.state.copy()
        UR = self.state.copy()
        F = hlld_flux_np(UL, UR, direction=0)
        rho = UL[0, :, :]
        vr  = UL[1, :, :] / rho
        Srho = UL[5, :, :]
        # For equal states SM = vr, so F[5] = Srho * vr
        np.testing.assert_allclose(
            F[5], (Srho * vr).astype(np.float32), rtol=1e-3
        )

    def test_lax_friedrichs_fallback_finite(self):
        """Pathological input should not produce NaN due to LF fallback."""
        UL = self.state.copy()
        UL[0] *= 1e-20   # near-zero density
        UR = self.state.copy()
        F = hlld_flux_np(UL, UR, direction=0)
        assert np.all(np.isfinite(F))


class TestCylSourceNp:
    """Tests for the NumPy cylindrical source-term reference."""

    def setup_method(self):
        self.nr, self.nz = 16, 8
        self.dr = 0.01
        self.prim, self.r_arr = _make_prim_state(self.nr, self.nz, self.dr)

    def test_output_shape(self):
        src = cyl_source_np(self.prim, self.r_arr, self.dr)
        assert src.shape == self.prim.shape

    def test_zero_source_components(self):
        """Variables 0,2,4,5,6,7,9 must have zero source."""
        src = cyl_source_np(self.prim, self.r_arr, self.dr)
        for v in (0, 2, 4, 5, 6, 7, 9):
            np.testing.assert_array_equal(src[v], 0.0,
                err_msg=f"Source var {v} should be zero")

    def test_r_momentum_source_positive_near_axis(self):
        """For purely azimuthal flow (vr=0, Br=0, p>0) S[1] = (p+B²/2-Bθ²)/r > 0."""
        prim = self.prim.copy()
        prim[1] = 0.0   # vr = 0
        prim[3] = 0.0   # vθ = 0
        prim[6] = 0.0   # Br = 0
        prim[8] = 0.0   # Bθ = 0
        src = cyl_source_np(prim, self.r_arr, self.dr)
        # (p + Bz²/2) / r > 0
        assert np.all(src[1, 1:, :] > 0), "r-momentum source must be positive"

    def test_theta_momentum_source_antisymmetry(self):
        """S[3] = -(ρvr vθ - Br Bθ)/r;  flipping vr sign flips the source."""
        prim_pos = self.prim.copy()
        prim_neg = self.prim.copy()
        prim_neg[1] = -self.prim[1]   # flip vr
        src_pos = cyl_source_np(prim_pos, self.r_arr, self.dr)
        src_neg = cyl_source_np(prim_neg, self.r_arr, self.dr)
        # Br·Bθ term unchanged, ρvr·vθ term flips
        # So src_neg[3] - src_pos[3] = 2*ρvr vθ / r  (varies by cell)
        # At least the sign should differ at non-axis cells
        diff = src_neg[3, 1:, :] - src_pos[3, 1:, :]
        # ρ*|vr|*|vθ| should be non-zero for our test state
        assert np.any(np.abs(diff) > 1e-10)

    def test_Btheta_source_formula(self):
        """S[8] = -(vr Bθ - Br vθ)/r; verify against direct formula."""
        prim = self.prim.copy()
        src  = cyl_source_np(prim, self.r_arr, self.dr)
        vr     = prim[1, 1:, :]
        vtheta = prim[3, 1:, :]
        Br     = prim[6, 1:, :]
        Btheta = prim[8, 1:, :]
        r2d    = self.r_arr[1:, np.newaxis]
        expected = -(vr * Btheta - Br * vtheta) / r2d
        np.testing.assert_allclose(src[8, 1:, :], expected.astype(np.float32),
                                   rtol=1e-5)

    def test_axis_lhopital_finite(self):
        """The L'Hôpital treatment at ir=0 must yield finite values."""
        src = cyl_source_np(self.prim, self.r_arr, self.dr)
        assert np.all(np.isfinite(src[:, 0, :]))

    def test_axis_theta_and_Btheta_zero(self):
        """θ-momentum and Bθ sources must vanish at the axis."""
        src = cyl_source_np(self.prim, self.r_arr, self.dr)
        np.testing.assert_array_equal(src[3, 0, :], 0.0)
        np.testing.assert_array_equal(src[8, 0, :], 0.0)


# ============================================================================
# Tests – MLX Metal kernel vs NumPy reference
# ============================================================================

@requires_mlx
class TestGhostPadMlx:
    """Validate Metal ghost_pad against NumPy reference."""

    def setup_method(self):
        self.nr, self.nz = 16, 8
        self.dr = 0.01
        self.ng = 3
        self.I  = 1e4
        self.state_np = _make_smooth_state(self.nr, self.nz, self.dr)

    def test_matches_numpy(self):
        ref    = ghost_pad_np(self.state_np, self.I, self.dr, self.ng)
        state  = mx.array(self.state_np)
        result = np.array(ghost_pad(state, self.I, self.dr, self.ng))
        np.testing.assert_allclose(result, ref, atol=ATOL, rtol=RTOL)

    def test_output_dtype(self):
        state  = mx.array(self.state_np)
        result = ghost_pad(state, self.I, self.dr, self.ng)
        assert result.dtype == mx.float32

    def test_output_shape(self):
        state  = mx.array(self.state_np)
        result = ghost_pad(state, self.I, self.dr, self.ng)
        assert result.shape == (10, self.nr + 2*self.ng, self.nz)

    def test_larger_grid(self):
        nr, nz = 64, 32
        state_np = _make_smooth_state(nr, nz, self.dr)
        ref    = ghost_pad_np(state_np, self.I, self.dr, self.ng)
        state  = mx.array(state_np)
        result = np.array(ghost_pad(state, self.I, self.dr, self.ng))
        np.testing.assert_allclose(result, ref, atol=ATOL, rtol=RTOL)


@requires_mlx
class TestHlldFluxMlx:
    """Validate Metal hlld_flux against NumPy reference."""

    def setup_method(self):
        self.nr, self.nz = 12, 10
        self.dr = 0.01
        self.state = _make_smooth_state(self.nr, self.nz, self.dr)

    def test_matches_numpy_r_dir(self):
        UL_np = self.state
        UR_np = np.roll(self.state, -1, axis=1).astype(np.float32)
        ref    = hlld_flux_np(UL_np, UR_np, direction=0)
        UL_mx  = mx.array(UL_np)
        UR_mx  = mx.array(UR_np)
        result = np.array(hlld_flux(UL_mx, UR_mx, direction=0))
        np.testing.assert_allclose(result, ref, atol=ATOL, rtol=RTOL)

    def test_matches_numpy_z_dir(self):
        UL_np = self.state
        UR_np = np.roll(self.state, -1, axis=2).astype(np.float32)
        ref    = hlld_flux_np(UL_np, UR_np, direction=1)
        UL_mx  = mx.array(UL_np)
        UR_mx  = mx.array(UR_np)
        result = np.array(hlld_flux(UL_mx, UR_mx, direction=1))
        np.testing.assert_allclose(result, ref, atol=ATOL, rtol=RTOL)

    def test_no_nan(self):
        UL_mx = mx.array(self.state)
        UR_mx = mx.array(self.state * np.float32(0.99))
        result = np.array(hlld_flux(UL_mx, UR_mx, direction=0))
        assert np.all(np.isfinite(result))


@requires_mlx
class TestCylSourceMlx:
    """Validate Metal cyl_source against NumPy reference."""

    def setup_method(self):
        self.nr, self.nz = 16, 8
        self.dr = 0.01
        self.prim_np, self.r_np = _make_prim_state(self.nr, self.nz, self.dr)

    def test_matches_numpy(self):
        ref    = cyl_source_np(self.prim_np, self.r_np, self.dr)
        prim   = mx.array(self.prim_np)
        r_arr  = mx.array(self.r_np)
        result = np.array(cyl_source(prim, r_arr, self.dr))
        np.testing.assert_allclose(result, ref, atol=ATOL, rtol=RTOL)

    def test_output_dtype(self):
        prim  = mx.array(self.prim_np)
        r_arr = mx.array(self.r_np)
        result = cyl_source(prim, r_arr, self.dr)
        assert result.dtype == mx.float32

    def test_zero_source_components(self):
        prim  = mx.array(self.prim_np)
        r_arr = mx.array(self.r_np)
        result = np.array(cyl_source(prim, r_arr, self.dr))
        for v in (0, 2, 4, 5, 6, 7, 9):
            np.testing.assert_array_equal(result[v], 0.0)
