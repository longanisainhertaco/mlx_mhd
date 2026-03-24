"""
mhd_kernels.py – Metal Shading Language kernels for cylindrical MHD solver.

Callable via ``mlx.fast.metal_kernel()`` on Apple Silicon (M-series).

State layout (SoA): float32 array of shape ``(10, nr, nz)``

Variable indices
----------------
  0  rho        mass density            [kg m⁻³]
  1  rho*vr     r-momentum density      [kg m⁻² s⁻¹]
  2  rho*vz     z-momentum density      [kg m⁻² s⁻¹]
  3  rho*vθ     θ-momentum density      [kg m⁻² s⁻¹]
  4  E          total energy density    [J m⁻³]
  5  Srho       entropy tracer ρ·s      [J m⁻³ K⁻¹]  (passive scalar)
  6  Br         radial magnetic field   [T]
  7  Bz         axial magnetic field    [T]
  8  Bθ         azimuthal B field       [T]
  9  Ee         electron energy density [J m⁻³]

Physical constants
------------------
  μ₀ = 4π × 10⁻⁷  H m⁻¹
  γ  = 5/3          (adiabatic index, deuterium)

Thread-group sizing for M3 Pro (14 GPU cores, SIMD width 32)
------------------------------------------------------------
  ghost_pad   : threadgroup = (1, 8, 8)  = 64 threads per group
  hlld_flux   : threadgroup = (8, 8, 1)  = 64 threads per group
  cyl_source  : threadgroup = (1, 8, 8)  = 64 threads per group

References
----------
  Miyoshi & Kusano (2005) – HLLD Riemann solver
  Cylindrical MHD geometric sources with L'Hôpital at r = 0
"""

from __future__ import annotations

import math
from typing import Sequence

# ---------------------------------------------------------------------------
# Optional MLX import – kernels degrade gracefully when MLX is unavailable
# (e.g. in CI without Apple Silicon).
# ---------------------------------------------------------------------------
try:
    import mlx.core as mx
    _HAS_MLX = True
except ImportError:  # pragma: no cover
    _HAS_MLX = False

# ---------------------------------------------------------------------------
# Physical / numerical constants
# ---------------------------------------------------------------------------
GAMMA: float = 5.0 / 3.0
MU0: float = 4.0 * math.pi * 1e-7
P_FLOOR: float = 1e-12
RHO_FLOOR: float = 1e-12

# ============================================================================
# Kernel 1 – Ghost Cell Padding with Electrode Boundary Conditions
# ============================================================================

_GHOST_PAD_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

/*
 * ghost_pad – pad a (10, nr, nz) state array to (10, nr+2*ng, nz) by
 * applying physics-based boundary conditions in the radial direction.
 *
 * Inner ghosts (r = 0 axis side):  reflecting BC
 *   sign-flip for rho*vr (var 1), Br (var 6), Bθ (var 8)
 *
 * Outer ghosts (cathode side):
 *   Bθ = μ₀ I / (2π r),  rho*vr = 0,  Br = 0,
 *   all others: zero-gradient (copy last interior cell)
 *
 * Buffers
 *   0  state   (10, nr, nz)     read-only input
 *   1  shape   [nr, nz, ng]     int32
 *   2  params  [I, dr]          float32
 *   3  padded  (10, nr+2ng, nz) write output
 *
 * Grid: (10, nr+2*ng, nz)  – one thread per (var, padded-r, z) cell
 */
[[kernel]] void ghost_pad(
    device const float* state   [[buffer(0)]],
    device const int*   shape   [[buffer(1)]],
    device const float* params  [[buffer(2)]],
    device       float* padded  [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]])
{
    const int ivar   = (int)tid.x;
    const int ir_pad = (int)tid.y;
    const int iz     = (int)tid.z;

    const int nr  = shape[0];
    const int nz  = shape[1];
    const int ng  = shape[2];
    const int nr_pad = nr + 2 * ng;

    if (ivar >= 10 || ir_pad >= nr_pad || iz >= nz) return;

    const float I_cur = params[0];
    const float dr    = params[1];

    const float MU0_F   = 4.0f * M_PI_F * 1.0e-7f;
    const float TWO_PI  = 2.0f * M_PI_F;

    const int out_idx = ivar * nr_pad * nz + ir_pad * nz + iz;

    if (ir_pad >= ng && ir_pad < ng + nr) {
        /* ---------- Interior: direct copy ---------- */
        int ir_src = ir_pad - ng;
        int in_idx = ivar * nr * nz + ir_src * nz + iz;
        padded[out_idx] = state[in_idx];

    } else if (ir_pad < ng) {
        /* ---------- Inner ghost (axis, reflecting BC) ----------
         * Ghost at ir_pad mirrors the interior cell at ir_src.
         *   ir_pad = ng-1  →  ir_src = 0   (nearest interior)
         *   ir_pad = ng-2  →  ir_src = 1
         *   ir_pad = 0     →  ir_src = ng-1
         */
        int ir_src = ng - 1 - ir_pad;
        int in_idx = ivar * nr * nz + ir_src * nz + iz;

        /* Sign flip for radial momentum, radial B, and azimuthal B */
        float sign = (ivar == 1 || ivar == 6 || ivar == 8) ? -1.0f : 1.0f;
        padded[out_idx] = sign * state[in_idx];

    } else {
        /* ---------- Outer ghost (cathode side) ----------
         * Cell-centre radius of padded cell ir_pad:
         *   r = (ir_pad - ng + 0.5) * dr
         * For outer ghosts ir_pad >= ng + nr, so r > domain max.
         */
        float r = ((float)(ir_pad - ng) + 0.5f) * dr;

        if (ivar == 8) {
            /* Bθ = μ₀ I / (2π r) */
            padded[out_idx] = MU0_F * I_cur / (TWO_PI * r);
        } else if (ivar == 1 || ivar == 6) {
            /* rho*vr = 0,  Br = 0 */
            padded[out_idx] = 0.0f;
        } else {
            /* Zero-gradient: copy last interior cell */
            int ir_src = nr - 1;
            int in_idx = ivar * nr * nz + ir_src * nz + iz;
            padded[out_idx] = state[in_idx];
        }
    }
}
"""


def ghost_pad(
    state: "mx.array",
    I: float,
    dr: float,
    ng: int = 3,
) -> "mx.array":
    """Apply ghost-cell padding with electrode boundary conditions.

    Parameters
    ----------
    state : mx.array, shape ``(10, nr, nz)``, float32
        Conserved state array.
    I : float
        Discharge current [A] used for the cathode Bθ BC.
    dr : float
        Radial cell spacing [m].
    ng : int
        Number of ghost cells on each radial side (default 3).

    Returns
    -------
    mx.array, shape ``(10, nr + 2*ng, nz)``, float32
    """
    if not _HAS_MLX:
        raise RuntimeError("MLX is not available in this environment.")
    assert state.ndim == 3 and state.shape[0] == 10, "state must be (10, nr, nz)"
    state = state.astype(mx.float32)
    nvars, nr, nz = state.shape
    nr_pad = nr + 2 * ng

    shape_arr  = mx.array([nr, nz, ng], dtype=mx.int32)
    params_arr = mx.array([float(I), float(dr)], dtype=mx.float32)

    kernel = mx.fast.metal_kernel(
        name="ghost_pad",
        input_names=["state", "shape", "params"],
        output_names=["padded"],
        source=_GHOST_PAD_SOURCE,
    )

    tg = (1, 8, 8)
    grid = (
        nvars,
        _ceil(nr_pad, tg[1]) * tg[1],
        _ceil(nz, tg[2]) * tg[2],
    )
    (padded,) = kernel(
        inputs=[state, shape_arr, params_arr],
        output_shapes=[(nvars, nr_pad, nz)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=tg,
    )
    return padded


# ============================================================================
# Kernel 2 – HLLD Riemann Solver (Miyoshi & Kusano 2005)
# ============================================================================

_HLLD_FLUX_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

/*
 * HLLD Riemann solver for cylindrical MHD.
 *
 * Each thread processes one cell interface and writes the 10-component
 * numerical flux.
 *
 * Variable ordering in UL / UR / flux (SoA layout):
 *   0 rho   1 rho*vr  2 rho*vz  3 rho*vθ  4 E
 *   5 Srho  6 Br      7 Bz      8 Bθ      9 Ee
 *
 * For direction dir:
 *   dir=0 (r-flux): normal=(r,1,6), t1=(z,2,7), t2=(θ,3,8)
 *   dir=1 (z-flux): normal=(z,2,7), t1=(r,1,6), t2=(θ,3,8)
 *
 * Buffers
 *   0  UL     (10, n_ir, n_iz)  left  reconstructed states
 *   1  UR     (10, n_ir, n_iz)  right reconstructed states
 *   2  iparams [n_ir, n_iz, dir]  int32
 *   3  flux   (10, n_ir, n_iz)  output numerical flux
 *
 * Grid: (n_ir, n_iz, 1)
 */

constant float HLLD_GAMMA    = 5.0f / 3.0f;
constant float HLLD_GAMMA_M1 = 2.0f / 3.0f;
constant float HLLD_P_FLOOR  = 1.0e-12f;
constant float HLLD_RHO_FLOOR = 1.0e-12f;
constant float HLLD_EPS      = 1.0e-12f;

/* ------------------------------------------------------------------ */
/* Extract primitives from conserved state buffer.                     */
/* Returns via thread-local references.                                */
/* ------------------------------------------------------------------ */
inline void hlld_primitives(
    device const float* U,
    int n_iface, int iface, int dir,
    thread float& rho,
    thread float& vn,  thread float& vt1, thread float& vt2,
    thread float& E,   thread float& Srho,
    thread float& Bn,  thread float& Bt1, thread float& Bt2, thread float& Ee,
    thread float& p,   thread float& ptot, thread float& vdotB)
{
    /* Momentum / B index mapping for each direction */
    const int i_mn  = 1 + dir;   /* normal momentum:   1 (r) or 2 (z) */
    const int i_mt1 = 2 - dir;   /* t1 momentum:       2 (r) or 1 (z) */
    const int i_mt2 = 3;         /* θ-momentum: always 3               */
    const int i_Bn  = 6 + dir;   /* normal B:          6 (r) or 7 (z) */
    const int i_Bt1 = 7 - dir;   /* t1 B:              7 (r) or 6 (z) */
    const int i_Bt2 = 8;         /* Bθ: always 8                       */

    rho  = max(U[0       * n_iface + iface], HLLD_RHO_FLOOR);
    E    = U[4           * n_iface + iface];
    Srho = U[5           * n_iface + iface];
    Bn   = U[i_Bn        * n_iface + iface];
    Bt1  = U[i_Bt1       * n_iface + iface];
    Bt2  = U[i_Bt2       * n_iface + iface];
    Ee   = U[9           * n_iface + iface];

    float inv_rho = 1.0f / rho;
    vn  = U[i_mn  * n_iface + iface] * inv_rho;
    vt1 = U[i_mt1 * n_iface + iface] * inv_rho;
    vt2 = U[i_mt2 * n_iface + iface] * inv_rho;

    float B2  = Bn*Bn + Bt1*Bt1 + Bt2*Bt2;
    float v2  = vn*vn + vt1*vt1 + vt2*vt2;
    vdotB     = vn*Bn + vt1*Bt1 + vt2*Bt2;

    p    = max(HLLD_GAMMA_M1 * (E - 0.5f*rho*v2 - 0.5f*B2), HLLD_P_FLOOR);
    ptot = p + 0.5f*B2;
}

/* ------------------------------------------------------------------ */
/* Fast magnetosonic speed (NaN-safe discriminant from problem spec).  */
/* cf² = ½(a²+va²+√D),  D = (a²−va²)² + 4a²Bt²/ρ  (always ≥ 0)    */
/* ------------------------------------------------------------------ */
inline float fast_speed(float rho, float p, float Bn,
                         float Bt1, float Bt2)
{
    float a2  = HLLD_GAMMA * p / rho;
    float B2  = Bn*Bn + Bt1*Bt1 + Bt2*Bt2;
    float va2 = B2 / rho;
    float Bt2_trans = Bt1*Bt1 + Bt2*Bt2;
    float disc = (a2 - va2)*(a2 - va2) + 4.0f*a2*Bt2_trans / rho;
    return sqrt(0.5f * (a2 + va2 + sqrt(max(disc, 0.0f))));
}

/* ------------------------------------------------------------------ */
/* Physical flux in the normal direction.                               */
/* Writes all 10 components into F[0..9] using the (in,t1,t2) layout. */
/* ------------------------------------------------------------------ */
inline void physical_flux(
    thread float* F,
    float rho, float vn, float vt1, float vt2,
    float E, float Srho, float Ee,
    float Bn, float Bt1, float Bt2,
    float p, float ptot, float vdotB,
    int i_mn, int i_mt1, int i_mt2,
    int i_Bn, int i_Bt1, int i_Bt2)
{
    float rho_vn = rho * vn;
    F[0]     = rho_vn;                            /* mass flux              */
    F[i_mn]  = rho_vn*vn + ptot - Bn*Bn;          /* normal momentum        */
    F[i_mt1] = rho_vn*vt1 - Bn*Bt1;               /* t1 momentum            */
    F[i_mt2] = rho_vn*vt2 - Bn*Bt2;               /* θ-momentum             */
    F[4]     = (E + ptot)*vn - Bn*vdotB;           /* energy                 */
    F[5]     = Srho * vn;                           /* entropy tracer         */
    F[i_Bn]  = 0.0f;                                /* normal B (no flux)     */
    F[i_Bt1] = vn*Bt1 - vt1*Bn;                    /* t1 B (induction)       */
    F[i_Bt2] = vn*Bt2 - vt2*Bn;                    /* θ B (induction)        */
    F[9]     = Ee * vn;                             /* electron energy        */
}

/* ------------------------------------------------------------------ */
/* Lax–Friedrichs fallback flux.                                       */
/* ------------------------------------------------------------------ */
inline void lax_friedrichs_flux(
    thread float* F,
    thread const float* FL, thread const float* FR,
    device const float* UL, device const float* UR,
    int n_iface, int iface,
    float S_max,
    int i_mn, int i_mt1, int i_mt2,
    int i_Bn, int i_Bt1, int i_Bt2)
{
    /* Iterate over all 10 variables; FL/FR store components by var index */
    for (int v = 0; v < 10; v++) {
        float ul = UL[v * n_iface + iface];
        float ur = UR[v * n_iface + iface];
        F[v] = 0.5f*(FL[v] + FR[v]) - 0.5f*S_max*(ur - ul);
    }
}

[[kernel]] void hlld_flux(
    device const float* UL      [[buffer(0)]],
    device const float* UR      [[buffer(1)]],
    device const int*   iparams [[buffer(2)]],
    device       float* flux    [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])
{
    const int ir  = (int)tid.x;
    const int iz  = (int)tid.y;

    const int n_ir  = iparams[0];
    const int n_iz  = iparams[1];
    const int dir   = iparams[2];

    if (ir >= n_ir || iz >= n_iz) return;

    const int n_iface = n_ir * n_iz;
    const int iface   = ir * n_iz + iz;

    /* Index mappings for this direction */
    const int i_mn  = 1 + dir;
    const int i_mt1 = 2 - dir;
    const int i_mt2 = 3;
    const int i_Bn  = 6 + dir;
    const int i_Bt1 = 7 - dir;
    const int i_Bt2 = 8;

    /* ---- Primitives ---- */
    float rhoL, vnL, vt1L, vt2L, EL, SrhoL, BnL, Bt1L, Bt2L, EeL, pL, ptotL, vdotBL;
    float rhoR, vnR, vt1R, vt2R, ER, SrhoR, BnR, Bt1R, Bt2R, EeR, pR, ptotR, vdotBR;

    hlld_primitives(UL, n_iface, iface, dir,
        rhoL, vnL, vt1L, vt2L, EL, SrhoL, BnL, Bt1L, Bt2L, EeL, pL, ptotL, vdotBL);
    hlld_primitives(UR, n_iface, iface, dir,
        rhoR, vnR, vt1R, vt2R, ER, SrhoR, BnR, Bt1R, Bt2R, EeR, pR, ptotR, vdotBR);

    /* Use the arithmetic-mean Bn for the star-region (CT consistency) */
    float Bn = 0.5f * (BnL + BnR);

    /* ---- Fast wave speeds ---- */
    float cfL = fast_speed(rhoL, pL, Bn, Bt1L, Bt2L);
    float cfR = fast_speed(rhoR, pR, Bn, Bt1R, Bt2R);

    /* ---- Outer wave speed estimates (Einfeldt / Davis) ---- */
    float cf_max = max(cfL, cfR);
    float SL = min(vnL, vnR) - cf_max;
    float SR = max(vnL, vnR) + cf_max;

    /* ---- Physical fluxes ---- */
    float FL[10], FR[10];
    physical_flux(FL, rhoL, vnL, vt1L, vt2L, EL, SrhoL, EeL,
                  Bn, Bt1L, Bt2L, pL, ptotL, vdotBL,
                  i_mn, i_mt1, i_mt2, i_Bn, i_Bt1, i_Bt2);
    physical_flux(FR, rhoR, vnR, vt1R, vt2R, ER, SrhoR, EeR,
                  Bn, Bt1R, Bt2R, pR, ptotR, vdotBR,
                  i_mn, i_mt1, i_mt2, i_Bn, i_Bt1, i_Bt2);

    float F[10];

    /* ---- Trivial cases ---- */
    if (SL >= 0.0f) {
        for (int v = 0; v < 10; v++) F[v] = FL[v];
        for (int v = 0; v < 10; v++) flux[v * n_iface + iface] = F[v];
        return;
    }
    if (SR <= 0.0f) {
        for (int v = 0; v < 10; v++) F[v] = FR[v];
        for (int v = 0; v < 10; v++) flux[v * n_iface + iface] = F[v];
        return;
    }

    /* ---- Contact speed and total pressure in star region ---- */
    float dL   = rhoL * (SL - vnL);    /* mass flux across SL (< 0) */
    float dR   = rhoR * (SR - vnR);    /* mass flux across SR (> 0) */
    float denom_M = dR - dL;           /* always > 0                 */

    float SM       = (dR*vnR - dL*vnL - ptotR + ptotL) / denom_M;
    float ptot_star = (dR*ptotL - dL*ptotR + dL*dR*(vnR - vnL)) / denom_M;

    /* ---- Star states K = L ---- */
    float rhoL_s = dL / (SL - SM);
    float denomL = rhoL*(SL-vnL)*(SL-vnL) - Bn*Bn;
    float vt1L_s, vt2L_s, Bt1L_s, Bt2L_s;
    if (fabs(denomL) < HLLD_EPS) {
        vt1L_s = vt1L;  vt2L_s = vt2L;
        Bt1L_s = Bt1L;  Bt2L_s = Bt2L;
    } else {
        float factv = Bn * (SM - vnL) / denomL;
        /* Bt* factor from M&K eq (45):                                    */
        /* Bt* = Bt * [ρ(SK-vn)(SK-SM) - Bn²] / [ρ(SK-vn)² - Bn²]        */
        float numL  = rhoL*(SL-vnL)*(SL-SM) - Bn*Bn;
        float BfacL = numL / denomL;
        vt1L_s = vt1L - Bt1L * factv;
        vt2L_s = vt2L - Bt2L * factv;
        Bt1L_s = Bt1L * BfacL;
        Bt2L_s = Bt2L * BfacL;
    }
    float vdotBL_s = SM*Bn + vt1L_s*Bt1L_s + vt2L_s*Bt2L_s;
    float EL_s = (EL*(SL-vnL) - ptotL*vnL + ptot_star*SM
                  + Bn*(vdotBL - vdotBL_s)) / (SL - SM);
    float SrhoL_s = SrhoL * (SL - vnL) / (SL - SM);
    float EeL_s   = EeL   * (SL - vnL) / (SL - SM);

    /* ---- Star states K = R ---- */
    float rhoR_s = dR / (SR - SM);
    float denomR = rhoR*(SR-vnR)*(SR-vnR) - Bn*Bn;
    float vt1R_s, vt2R_s, Bt1R_s, Bt2R_s;
    if (fabs(denomR) < HLLD_EPS) {
        vt1R_s = vt1R;  vt2R_s = vt2R;
        Bt1R_s = Bt1R;  Bt2R_s = Bt2R;
    } else {
        float factv = Bn * (SM - vnR) / denomR;
        float numR  = rhoR*(SR-vnR)*(SR-SM) - Bn*Bn;
        float BfacR = numR / denomR;
        vt1R_s = vt1R - Bt1R * factv;
        vt2R_s = vt2R - Bt2R * factv;
        Bt1R_s = Bt1R * BfacR;
        Bt2R_s = Bt2R * BfacR;
    }
    float vdotBR_s = SM*Bn + vt1R_s*Bt1R_s + vt2R_s*Bt2R_s;
    float ER_s = (ER*(SR-vnR) - ptotR*vnR + ptot_star*SM
                  + Bn*(vdotBR - vdotBR_s)) / (SR - SM);
    float SrhoR_s = SrhoR * (SR - vnR) / (SR - SM);
    float EeR_s   = EeR   * (SR - vnR) / (SR - SM);

    /* ---- Alfvén wave speeds ---- */
    float sqrtRhoL_s = sqrt(max(rhoL_s, HLLD_RHO_FLOOR));
    float sqrtRhoR_s = sqrt(max(rhoR_s, HLLD_RHO_FLOOR));
    float SA_L = SM - fabs(Bn) / sqrtRhoL_s;
    float SA_R = SM + fabs(Bn) / sqrtRhoR_s;

    /* ---- Double-star states (between the two Alfvén waves) ---- */
    float signBn     = (Bn >= 0.0f) ? 1.0f : -1.0f;
    float denom_dd   = sqrtRhoL_s + sqrtRhoR_s;
    float inv_dd     = 1.0f / max(denom_dd, HLLD_EPS);

    float vt1_dd = (sqrtRhoL_s*vt1L_s + sqrtRhoR_s*vt1R_s
                    + signBn*(Bt1R_s - Bt1L_s)) * inv_dd;
    float vt2_dd = (sqrtRhoL_s*vt2L_s + sqrtRhoR_s*vt2R_s
                    + signBn*(Bt2R_s - Bt2L_s)) * inv_dd;
    float Bt1_dd = (sqrtRhoL_s*Bt1R_s + sqrtRhoR_s*Bt1L_s
                    + signBn*sqrtRhoL_s*sqrtRhoR_s*(vt1R_s - vt1L_s)) * inv_dd;
    float Bt2_dd = (sqrtRhoL_s*Bt2R_s + sqrtRhoR_s*Bt2L_s
                    + signBn*sqrtRhoL_s*sqrtRhoR_s*(vt2R_s - vt2L_s)) * inv_dd;
    float vdotB_dd = SM*Bn + vt1_dd*Bt1_dd + vt2_dd*Bt2_dd;

    float EL_dd = EL_s - signBn*sqrtRhoL_s*(vdotBL_s - vdotB_dd);
    float ER_dd = ER_s + signBn*sqrtRhoR_s*(vdotBR_s - vdotB_dd);
    /* Density and entropy are unchanged across Alfvén waves */
    float rhoL_dd = rhoL_s,  SrhoL_dd = SrhoL_s, EeL_dd = EeL_s;
    float rhoR_dd = rhoR_s,  SrhoR_dd = SrhoR_s, EeR_dd = EeR_s;

    /* ---- Star fluxes using Rankine–Hugoniot ----
     * F*K = FK + SK*(U*K - UK)                                     */
    float FL_s[10], FR_s[10], FL_dd[10], FR_dd[10];

    /* Repack star states into var-index order for convenience */
    /* Left star */
    float UL_s[10];
    UL_s[0]     = rhoL_s;
    UL_s[i_mn]  = rhoL_s * SM;
    UL_s[i_mt1] = rhoL_s * vt1L_s;
    UL_s[i_mt2] = rhoL_s * vt2L_s;
    UL_s[4]     = EL_s;
    UL_s[5]     = SrhoL_s;
    UL_s[i_Bn]  = Bn;
    UL_s[i_Bt1] = Bt1L_s;
    UL_s[i_Bt2] = Bt2L_s;
    UL_s[9]     = EeL_s;

    /* Right star */
    float UR_s[10];
    UR_s[0]     = rhoR_s;
    UR_s[i_mn]  = rhoR_s * SM;
    UR_s[i_mt1] = rhoR_s * vt1R_s;
    UR_s[i_mt2] = rhoR_s * vt2R_s;
    UR_s[4]     = ER_s;
    UR_s[5]     = SrhoR_s;
    UR_s[i_Bn]  = Bn;
    UR_s[i_Bt1] = Bt1R_s;
    UR_s[i_Bt2] = Bt2R_s;
    UR_s[9]     = EeR_s;

    /* Left double-star */
    float UL_dd[10];
    UL_dd[0]     = rhoL_dd;
    UL_dd[i_mn]  = rhoL_dd * SM;
    UL_dd[i_mt1] = rhoL_dd * vt1_dd;
    UL_dd[i_mt2] = rhoL_dd * vt2_dd;
    UL_dd[4]     = EL_dd;
    UL_dd[5]     = SrhoL_dd;
    UL_dd[i_Bn]  = Bn;
    UL_dd[i_Bt1] = Bt1_dd;
    UL_dd[i_Bt2] = Bt2_dd;
    UL_dd[9]     = EeL_dd;

    /* Right double-star */
    float UR_dd[10];
    UR_dd[0]     = rhoR_dd;
    UR_dd[i_mn]  = rhoR_dd * SM;
    UR_dd[i_mt1] = rhoR_dd * vt1_dd;
    UR_dd[i_mt2] = rhoR_dd * vt2_dd;
    UR_dd[4]     = ER_dd;
    UR_dd[5]     = SrhoR_dd;
    UR_dd[i_Bn]  = Bn;
    UR_dd[i_Bt1] = Bt1_dd;
    UR_dd[i_Bt2] = Bt2_dd;
    UR_dd[9]     = EeR_dd;

    /* F*K  = FK + SK * (U*K - UK)  (read UK from buffer) */
    for (int v = 0; v < 10; v++) {
        float ul = UL[v * n_iface + iface];
        float ur = UR[v * n_iface + iface];
        FL_s[v]  = FL[v]  + SL * (UL_s[v]  - ul);
        FR_s[v]  = FR[v]  + SR * (UR_s[v]  - ur);
        FL_dd[v] = FL_s[v] + SA_L * (UL_dd[v] - UL_s[v]);
        FR_dd[v] = FR_s[v] + SA_R * (UR_dd[v] - UR_s[v]);
    }

    /* ---- Override entropy component to advect at contact speed SM ----
     * The entropy tracer is a passive scalar on the contact discontinuity.
     * F[5] = ρ*s upwinded to SM.                                        */
    float Srho_contact = (SM >= 0.0f) ? SrhoL_s : SrhoR_s;
    float Srho_contact_dd = (SM >= 0.0f) ? SrhoL_dd : SrhoR_dd;
    FL_s[5]  = Srho_contact    * SM;
    FR_s[5]  = Srho_contact    * SM;
    FL_dd[5] = Srho_contact_dd * SM;
    FR_dd[5] = Srho_contact_dd * SM;

    /* ---- Region selection ---- */
    if (SA_L >= 0.0f) {
        /* SL < 0 ≤ SA_L : use FL* */
        for (int v = 0; v < 10; v++) F[v] = FL_s[v];
    } else if (SM >= 0.0f) {
        /* SA_L < 0 ≤ SM : use FL** */
        for (int v = 0; v < 10; v++) F[v] = FL_dd[v];
    } else if (SA_R > 0.0f) {
        /* SM < 0 < SA_R : use FR** */
        for (int v = 0; v < 10; v++) F[v] = FR_dd[v];
    } else {
        /* SA_R ≤ 0 < SR : use FR* */
        for (int v = 0; v < 10; v++) F[v] = FR_s[v];
    }

    /* ---- NaN / Inf safety: fall back to Lax–Friedrichs ---- */
    float check = 0.0f;
    for (int v = 0; v < 10; v++) check += F[v];
    if (isnan(check) || isinf(check)) {
        float S_max = max(fabs(SL), fabs(SR));
        for (int v = 0; v < 10; v++) {
            float ul = UL[v * n_iface + iface];
            float ur = UR[v * n_iface + iface];
            F[v] = 0.5f*(FL[v] + FR[v]) - 0.5f*S_max*(ur - ul);
        }
    }

    for (int v = 0; v < 10; v++)
        flux[v * n_iface + iface] = F[v];
}
"""


def hlld_flux(
    UL: "mx.array",
    UR: "mx.array",
    direction: int,
) -> "mx.array":
    """Compute HLLD numerical flux at cell interfaces.

    Parameters
    ----------
    UL : mx.array, shape ``(10, n_ir, n_iz)``, float32
        Left reconstructed conserved states at interfaces.
    UR : mx.array, shape ``(10, n_ir, n_iz)``, float32
        Right reconstructed conserved states at interfaces.
    direction : int
        0 → r-direction flux,  1 → z-direction flux.

    Returns
    -------
    mx.array, shape ``(10, n_ir, n_iz)``, float32
        Numerical flux vector at each interface.
    """
    if not _HAS_MLX:
        raise RuntimeError("MLX is not available in this environment.")
    assert UL.ndim == 3 and UL.shape[0] == 10, "UL must be (10, n_ir, n_iz)"
    assert UR.shape == UL.shape, "UL and UR must have the same shape"
    UL = UL.astype(mx.float32)
    UR = UR.astype(mx.float32)
    _, n_ir, n_iz = UL.shape

    iparams = mx.array([n_ir, n_iz, int(direction)], dtype=mx.int32)

    kernel = mx.fast.metal_kernel(
        name="hlld_flux",
        input_names=["UL", "UR", "iparams"],
        output_names=["flux"],
        source=_HLLD_FLUX_SOURCE,
    )

    tg = (8, 8, 1)
    grid = (
        _ceil(n_ir, tg[0]) * tg[0],
        _ceil(n_iz, tg[1]) * tg[1],
        1,
    )
    (out,) = kernel(
        inputs=[UL, UR, iparams],
        output_shapes=[(10, n_ir, n_iz)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=tg,
    )
    return out


# ============================================================================
# Kernel 3 – Cylindrical Geometric Source Terms
# ============================================================================

_CYL_SOURCE_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

/*
 * cyl_source – geometric source terms for cylindrical MHD.
 *
 * Input is the *primitive* state (10 components, same SoA layout):
 *   0 rho   1 vr   2 vz   3 vθ   4 p
 *   5 S     6 Br   7 Bz   8 Bθ   9 Ee
 *
 * Source terms:
 *   S[0]  = 0
 *   S[1]  = (p + B²/2 - Bθ²)/r + ρvθ²/r      (r-momentum)
 *   S[2]  = 0
 *   S[3]  = -(ρvr vθ - Br Bθ)/r               (θ-momentum)
 *   S[4]  = 0
 *   S[5]  = 0
 *   S[6]  = 0
 *   S[7]  = 0
 *   S[8]  = -(vr Bθ - Br vθ)/r                (Bθ)
 *   S[9]  = 0
 *
 * At r → 0 (ir = 0): L'Hôpital – replace 1/r terms with d(·)/dr
 * approximated as a first-order one-sided difference using ir=1.
 *
 * Buffers
 *   0  prim    (10, nr, nz)   primitive state (read-only)
 *   1  r_arr   (nr,)          cell-centre radii [m] (read-only)
 *   2  fparams [dr]           float32
 *   3  src     (10, nr, nz)   output source terms
 *
 * Grid: (10, nr, nz)
 */

[[kernel]] void cyl_source(
    device const float* prim    [[buffer(0)]],
    device const float* r_arr   [[buffer(1)]],
    device const float* fparams [[buffer(2)]],
    device       float* src     [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]])
{
    const int ivar = (int)tid.x;
    const int ir   = (int)tid.y;
    const int iz   = (int)tid.z;

    /* shape is embedded implicitly; threads outside the grid do nothing */
    /* We learn nr and nz from the grid bounds, but we need them from params */
    /* Instead, check thread validity via the source pointer stride.
     * Since we don't have nr/nz here, the Python wrapper must ensure the
     * grid equals (10, nr, nz) exactly so there are no out-of-bounds. */

    /* Zero-source components: 0 (rho), 2 (vz), 4 (E), 5 (S), 6 (Br),
     *                          7 (Bz), 9 (Ee)                           */
    /* We still need nr and nz to compute the linear index. Pass them.    */
    /* NOTE: nr and nz are obtained from iparams; we reuse fparams[1..2] */
    /* Revised: fparams = [dr, nr_f, nz_f] where last two cast to int.   */
    const float dr = fparams[0];
    const int   nr = (int)fparams[1];
    const int   nz = (int)fparams[2];

    if (ivar >= 10 || ir >= nr || iz >= nz) return;

    const int stride = nr * nz;
    const int idx    = ivar * stride + ir * nz + iz;

    /* Zero-source variables */
    if (ivar == 0 || ivar == 2 || ivar == 4 ||
        ivar == 5 || ivar == 6 || ivar == 7 || ivar == 9) {
        src[idx] = 0.0f;
        return;
    }

    /* Load primitive state at (ir, iz) */
    auto prim_at = [&](int v, int row, int col) -> float {
        return prim[v * stride + row * nz + col];
    };

    float rho    = prim_at(0, ir, iz);
    float vr     = prim_at(1, ir, iz);
    float vtheta = prim_at(3, ir, iz);
    float p      = prim_at(4, ir, iz);
    float Br     = prim_at(6, ir, iz);
    float Bz     = prim_at(7, ir, iz);
    float Btheta = prim_at(8, ir, iz);
    float B2     = Br*Br + Bz*Bz + Btheta*Btheta;

    float result = 0.0f;

    if (ir == 0) {
        /* ---- L'Hôpital at r = 0: replace f/r with df/dr ---- */
        if (ivar == 1) {
            /* d(p + B²/2 - Bθ²)/dr + d(ρvθ²)/dr at ir=0          */
            /* Approximate with one-sided difference to ir=1        */
            float rho1    = prim_at(0, 1, iz);
            float vtheta1 = prim_at(3, 1, iz);
            float p1      = prim_at(4, 1, iz);
            float Br1     = prim_at(6, 1, iz);
            float Bz1     = prim_at(7, 1, iz);
            float Btheta1 = prim_at(8, 1, iz);
            float B2_1    = Br1*Br1 + Bz1*Bz1 + Btheta1*Btheta1;
            float f0 = p      + 0.5f*B2   - Btheta*Btheta   + rho   *vtheta  *vtheta;
            float f1 = p1     + 0.5f*B2_1 - Btheta1*Btheta1 + rho1  *vtheta1 *vtheta1;
            result = (f1 - f0) / dr;
        } else if (ivar == 3) {
            /* d(-ρvr vθ + Br Bθ)/dr at r=0; vr=0 and Br=0 at axis → 0 */
            result = 0.0f;
        } else if (ivar == 8) {
            /* d(-(vr Bθ - Br vθ))/dr at r=0; vr=0, Br=0 at axis → 0 */
            result = 0.0f;
        }
    } else {
        /* ---- Regular cells: use cell-centre radius ---- */
        float r   = r_arr[ir];
        float inv_r = 1.0f / r;

        if (ivar == 1) {
            /* (p + B²/2 - Bθ²)/r + ρvθ²/r */
            result = (p + 0.5f*B2 - Btheta*Btheta) * inv_r
                   + rho * vtheta * vtheta * inv_r;
        } else if (ivar == 3) {
            /* -(ρvr vθ - Br Bθ)/r */
            result = -(rho*vr*vtheta - Br*Btheta) * inv_r;
        } else if (ivar == 8) {
            /* -(vr Bθ - Br vθ)/r */
            result = -(vr*Btheta - Br*vtheta) * inv_r;
        }
    }

    src[idx] = result;
}
"""


def cyl_source(
    prim: "mx.array",
    r_arr: "mx.array",
    dr: float,
) -> "mx.array":
    """Compute cylindrical geometric source terms.

    Parameters
    ----------
    prim : mx.array, shape ``(10, nr, nz)``, float32
        Primitive state:
        ``[rho, vr, vz, vθ, p, S, Br, Bz, Bθ, Ee]``
    r_arr : mx.array, shape ``(nr,)``, float32
        Cell-centre radii [m].
    dr : float
        Radial cell spacing [m] (used for L'Hôpital at r = 0).

    Returns
    -------
    mx.array, shape ``(10, nr, nz)``, float32
        Geometric source term array.
    """
    if not _HAS_MLX:
        raise RuntimeError("MLX is not available in this environment.")
    assert prim.ndim == 3 and prim.shape[0] == 10, "prim must be (10, nr, nz)"
    prim  = prim.astype(mx.float32)
    r_arr = r_arr.astype(mx.float32)
    nvars, nr, nz = prim.shape
    assert r_arr.shape == (nr,), "r_arr must have shape (nr,)"

    # fparams = [dr, nr (as float), nz (as float)]
    fparams = mx.array([float(dr), float(nr), float(nz)], dtype=mx.float32)

    kernel = mx.fast.metal_kernel(
        name="cyl_source",
        input_names=["prim", "r_arr", "fparams"],
        output_names=["src"],
        source=_CYL_SOURCE_SOURCE,
    )

    tg = (1, 8, 8)
    grid = (
        nvars,
        _ceil(nr, tg[1]) * tg[1],
        _ceil(nz, tg[2]) * tg[2],
    )
    (out,) = kernel(
        inputs=[prim, r_arr, fparams],
        output_shapes=[(nvars, nr, nz)],
        output_dtypes=[mx.float32],
        grid=grid,
        threadgroup=tg,
    )
    return out


# ============================================================================
# Utility
# ============================================================================

def _ceil(n: int, d: int) -> int:
    """Return ⌈n/d⌉."""
    return (n + d - 1) // d
