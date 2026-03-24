"""
mhd_reference.py – NumPy reference implementations for the three MHD kernels.

These are intentionally simple and unoptimised – they serve as a ground-truth
baseline against which the Metal GPU kernels are validated.

Variable ordering (SoA, same as mhd_kernels.py):
  0 rho   1 rho*vr  2 rho*vz  3 rho*vθ  4 E
  5 Srho  6 Br      7 Bz      8 Bθ      9 Ee
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Physical / numerical constants
# ---------------------------------------------------------------------------
GAMMA: float = 5.0 / 3.0
GAMMA_M1: float = GAMMA - 1.0
MU0: float = 4.0 * math.pi * 1e-7
P_FLOOR: float = 1e-12
RHO_FLOOR: float = 1e-12


# ============================================================================
# Kernel 1 reference – Ghost Cell Padding
# ============================================================================

def ghost_pad_np(
    state: np.ndarray,
    I: float,
    dr: float,
    ng: int = 3,
) -> np.ndarray:
    """NumPy reference for ghost-cell padding.

    Parameters
    ----------
    state : ndarray, shape (10, nr, nz), float32
        Conserved state.
    I : float
        Discharge current [A].
    dr : float
        Radial cell spacing [m].
    ng : int
        Ghost-cell count (default 3).

    Returns
    -------
    ndarray, shape (10, nr+2*ng, nz), float32
    """
    assert state.ndim == 3 and state.shape[0] == 10
    state = state.astype(np.float32)
    nvars, nr, nz = state.shape
    nr_pad = nr + 2 * ng

    padded = np.zeros((nvars, nr_pad, nz), dtype=np.float32)

    # ---- Interior ----
    padded[:, ng : ng + nr, :] = state

    # ---- Inner ghosts (axis, reflecting BC) ----
    for g in range(ng):
        ir_pad = ng - 1 - g          # ghost index in padded array (0..ng-1)
        ir_src = g                   # mirrored interior index
        padded[:, ir_pad, :] = state[:, ir_src, :]
        # Sign flip for rho*vr (1), Br (6), Bθ (8)
        for v in (1, 6, 8):
            padded[v, ir_pad, :] = -state[v, ir_src, :]

    # ---- Outer ghosts (cathode side) ----
    for g in range(ng):
        ir_pad = ng + nr + g
        r = (ir_pad - ng + 0.5) * dr   # cell-centre radius of ghost cell
        # Zero-gradient default
        padded[:, ir_pad, :] = state[:, nr - 1, :]
        # vr = 0, Br = 0
        padded[1, ir_pad, :] = 0.0
        padded[6, ir_pad, :] = 0.0
        # Bθ = μ₀ I / (2π r)
        padded[8, ir_pad, :] = MU0 * I / (2.0 * math.pi * r)

    return padded


# ============================================================================
# Kernel 2 reference – HLLD Riemann Solver (Miyoshi & Kusano 2005)
# ============================================================================

def _primitives(U: np.ndarray) -> dict:
    """Extract primitive variables from a conserved state vector (length 10)."""
    rho = max(float(U[0]), RHO_FLOOR)
    inv = 1.0 / rho
    vr  = float(U[1]) * inv
    vz  = float(U[2]) * inv
    vth = float(U[3]) * inv
    E   = float(U[4])
    Srho = float(U[5])
    Br  = float(U[6])
    Bz  = float(U[7])
    Bth = float(U[8])
    Ee  = float(U[9])
    B2  = Br*Br + Bz*Bz + Bth*Bth
    v2  = vr*vr + vz*vz + vth*vth
    p   = max(GAMMA_M1 * (E - 0.5*rho*v2 - 0.5*B2), P_FLOOR)
    ptot = p + 0.5 * B2
    vdotB = vr*Br + vz*Bz + vth*Bth
    return dict(rho=rho, vr=vr, vz=vz, vth=vth,
                E=E, Srho=Srho, Ee=Ee,
                Br=Br, Bz=Bz, Bth=Bth,
                p=p, ptot=ptot, vdotB=vdotB, B2=B2, v2=v2)


def _fast_speed(rho: float, p: float, Bn: float, Bt1: float, Bt2: float) -> float:
    """Fast magnetosonic speed (NaN-safe discriminant)."""
    a2  = GAMMA * p / rho
    B2  = Bn*Bn + Bt1*Bt1 + Bt2*Bt2
    va2 = B2 / rho
    Bt2_trans = Bt1*Bt1 + Bt2*Bt2
    disc = (a2 - va2)**2 + 4.0 * a2 * Bt2_trans / rho
    return math.sqrt(0.5 * (a2 + va2 + math.sqrt(max(disc, 0.0))))


def _physical_flux_1d(prim: dict, Bn: float, Bt1: float, Bt2: float,
                       i_mn: int, i_mt1: int, i_mt2: int,
                       i_Bn: int, i_Bt1: int, i_Bt2: int) -> np.ndarray:
    """Physical flux vector in the normal direction for a 1-D state."""
    rho   = prim["rho"]
    vn    = prim["vr"] if i_mn == 1 else prim["vz"]
    vt1   = prim["vz"] if i_mt1 == 2 else prim["vr"]
    vt2   = prim["vth"]
    E     = prim["E"]
    Srho  = prim["Srho"]
    Ee    = prim["Ee"]
    ptot  = prim["ptot"]
    vdotB = vn*Bn + vt1*Bt1 + vt2*Bt2

    F = np.zeros(10, dtype=np.float64)
    rho_vn    = rho * vn
    F[0]      = rho_vn
    F[i_mn]   = rho_vn*vn + ptot - Bn*Bn
    F[i_mt1]  = rho_vn*vt1 - Bn*Bt1
    F[i_mt2]  = rho_vn*vt2 - Bn*Bt2
    F[4]      = (E + ptot)*vn - Bn*vdotB
    F[5]      = Srho * vn
    F[i_Bn]   = 0.0
    F[i_Bt1]  = vn*Bt1 - vt1*Bn
    F[i_Bt2]  = vn*Bt2 - vt2*Bn
    F[9]      = Ee * vn
    return F


def _hlld_1d(UL: np.ndarray, UR: np.ndarray, direction: int) -> np.ndarray:
    """HLLD flux for a single interface (length-10 state vectors)."""
    assert UL.shape == (10,) and UR.shape == (10,)

    # Index mapping
    i_mn  = 1 + direction
    i_mt1 = 2 - direction
    i_mt2 = 3
    i_Bn  = 6 + direction
    i_Bt1 = 7 - direction
    i_Bt2 = 8

    primL = _primitives(UL)
    primR = _primitives(UR)

    rhoL = primL["rho"];  rhoR = primR["rho"]
    vnL  = primL["vr"] if i_mn == 1 else primL["vz"]
    vnR  = primR["vr"] if i_mn == 1 else primR["vz"]
    vt1L = primL["vz"] if i_mt1 == 2 else primL["vr"]
    vt1R = primR["vz"] if i_mt1 == 2 else primR["vr"]
    vt2L = primL["vth"];  vt2R = primR["vth"]

    # Use arithmetic-mean Bn (CT consistency)
    Bn   = 0.5 * (UL[i_Bn] + UR[i_Bn])
    Bt1L = UL[i_Bt1]; Bt2L = UL[i_Bt2]
    Bt1R = UR[i_Bt1]; Bt2R = UR[i_Bt2]

    EL = primL["E"];   ER = primR["E"]
    pL = primL["p"];   pR = primR["p"]
    ptotL = primL["ptot"]; ptotR = primR["ptot"]
    vdotBL = primL["vdotB"]; vdotBR = primR["vdotB"]
    SrhoL = UL[5]; SrhoR = UR[5]
    EeL   = UL[9]; EeR   = UR[9]

    cfL = _fast_speed(rhoL, pL, Bn, Bt1L, Bt2L)
    cfR = _fast_speed(rhoR, pR, Bn, Bt1R, Bt2R)

    cf_max = max(cfL, cfR)
    SL = min(vnL, vnR) - cf_max
    SR = max(vnL, vnR) + cf_max

    FL = _physical_flux_1d(primL, Bn, Bt1L, Bt2L,
                            i_mn, i_mt1, i_mt2, i_Bn, i_Bt1, i_Bt2)
    FR = _physical_flux_1d(primR, Bn, Bt1R, Bt2R,
                            i_mn, i_mt1, i_mt2, i_Bn, i_Bt1, i_Bt2)

    if SL >= 0.0:
        return FL
    if SR <= 0.0:
        return FR

    dL = rhoL * (SL - vnL)
    dR = rhoR * (SR - vnR)
    denom_M = dR - dL

    SM       = (dR*vnR - dL*vnL - ptotR + ptotL) / denom_M
    ptot_star = (dR*ptotL - dL*ptotR + dL*dR*(vnR - vnL)) / denom_M

    eps = 1e-12

    class _StarState(NamedTuple):
        rho: float
        vn: float
        vt1: float
        vt2: float
        E: float
        Srho: float
        Bt1: float
        Bt2: float
        vdotB: float
        Ee: float

    def _star_state(rho, vn, vt1, vt2, E, Srho, Ee,
                     Bt1, Bt2, ptot, vdotB_orig, SK) -> _StarState:
        dK     = SK - vn
        gK     = rho * dK
        denomK = rho * dK * dK - Bn * Bn

        rho_s = gK / (SK - SM)
        if abs(denomK) < eps:
            vt1_s, vt2_s = vt1, vt2
            Bt1_s, Bt2_s = Bt1, Bt2
        else:
            factv  = Bn * (SM - vn) / denomK
            numK   = rho * dK * (SK - SM) - Bn * Bn
            BfacK  = numK / denomK
            vt1_s  = vt1 - Bt1 * factv
            vt2_s  = vt2 - Bt2 * factv
            Bt1_s  = Bt1 * BfacK
            Bt2_s  = Bt2 * BfacK

        vdotB_s = SM * Bn + vt1_s * Bt1_s + vt2_s * Bt2_s
        E_s     = (E * dK - ptot*vn + ptot_star*SM
                   + Bn*(vdotB_orig - vdotB_s)) / (SK - SM)
        Srho_s  = Srho * dK / (SK - SM)
        Ee_s    = Ee   * dK / (SK - SM)
        return _StarState(rho_s, SM, vt1_s, vt2_s, E_s, Srho_s,
                          Bt1_s, Bt2_s, vdotB_s, Ee_s)

    ssL = _star_state(rhoL, vnL, vt1L, vt2L, EL, SrhoL, EeL,
                      Bt1L, Bt2L, ptotL, vdotBL, SL)
    ssR = _star_state(rhoR, vnR, vt1R, vt2R, ER, SrhoR, EeR,
                      Bt1R, Bt2R, ptotR, vdotBR, SR)

    rhoL_s = ssL.rho;  vt1L_s = ssL.vt1;  vt2L_s = ssL.vt2
    EL_s   = ssL.E;    SrhoL_s = ssL.Srho; EeL_s  = ssL.Ee
    Bt1L_s = ssL.Bt1;  Bt2L_s  = ssL.Bt2;  vdotBL_s = ssL.vdotB

    rhoR_s = ssR.rho;  vt1R_s = ssR.vt1;  vt2R_s = ssR.vt2
    ER_s   = ssR.E;    SrhoR_s = ssR.Srho; EeR_s  = ssR.Ee
    Bt1R_s = ssR.Bt1;  Bt2R_s  = ssR.Bt2;  vdotBR_s = ssR.vdotB

    sqrtRhoL_s = math.sqrt(max(rhoL_s, eps))
    sqrtRhoR_s = math.sqrt(max(rhoR_s, eps))
    SA_L = SM - abs(Bn) / sqrtRhoL_s
    SA_R = SM + abs(Bn) / sqrtRhoR_s

    # Double-star states
    signBn   = 1.0 if Bn >= 0.0 else -1.0
    denom_dd = sqrtRhoL_s + sqrtRhoR_s
    if abs(denom_dd) < eps:
        denom_dd = eps

    vt1_dd = (sqrtRhoL_s*vt1L_s + sqrtRhoR_s*vt1R_s
              + signBn*(Bt1R_s - Bt1L_s)) / denom_dd
    vt2_dd = (sqrtRhoL_s*vt2L_s + sqrtRhoR_s*vt2R_s
              + signBn*(Bt2R_s - Bt2L_s)) / denom_dd
    Bt1_dd = (sqrtRhoL_s*Bt1R_s + sqrtRhoR_s*Bt1L_s
              + signBn*sqrtRhoL_s*sqrtRhoR_s*(vt1R_s - vt1L_s)) / denom_dd
    Bt2_dd = (sqrtRhoL_s*Bt2R_s + sqrtRhoR_s*Bt2L_s
              + signBn*sqrtRhoL_s*sqrtRhoR_s*(vt2R_s - vt2L_s)) / denom_dd
    vdotB_dd = SM*Bn + vt1_dd*Bt1_dd + vt2_dd*Bt2_dd

    EL_dd = EL_s - signBn*sqrtRhoL_s*(vdotBL_s - vdotB_dd)
    ER_dd = ER_s + signBn*sqrtRhoR_s*(vdotBR_s - vdotB_dd)

    # Pack star / double-star states into conserved vectors
    def _pack(rho_s, vn_s, vt1_s, vt2_s, E_s, Srho_s, Ee_s,
               Bt1_s, Bt2_s):
        U_s = np.zeros(10)
        U_s[0]     = rho_s
        U_s[i_mn]  = rho_s * vn_s
        U_s[i_mt1] = rho_s * vt1_s
        U_s[i_mt2] = rho_s * vt2_s
        U_s[4]     = E_s
        U_s[5]     = Srho_s
        U_s[i_Bn]  = Bn
        U_s[i_Bt1] = Bt1_s
        U_s[i_Bt2] = Bt2_s
        U_s[9]     = Ee_s
        return U_s

    UL_s  = _pack(rhoL_s, SM, vt1L_s, vt2L_s, EL_s,  SrhoL_s, EeL_s,  Bt1L_s, Bt2L_s)
    UR_s  = _pack(rhoR_s, SM, vt1R_s, vt2R_s, ER_s,  SrhoR_s, EeR_s,  Bt1R_s, Bt2R_s)
    UL_dd = _pack(rhoL_s, SM, vt1_dd, vt2_dd, EL_dd, SrhoL_s, EeL_s,  Bt1_dd, Bt2_dd)
    UR_dd = _pack(rhoR_s, SM, vt1_dd, vt2_dd, ER_dd, SrhoR_s, EeR_s,  Bt1_dd, Bt2_dd)

    FL_s  = FL  + SL   * (UL_s  - UL)
    FR_s  = FR  + SR   * (UR_s  - UR)
    FL_dd = FL_s + SA_L * (UL_dd - UL_s)
    FR_dd = FR_s + SA_R * (UR_dd - UR_s)

    # Override entropy component to be advected by contact speed SM
    Srho_contact = SrhoL_s if SM >= 0.0 else SrhoR_s
    for Fv in (FL_s, FR_s, FL_dd, FR_dd):
        Fv[5] = Srho_contact * SM

    # Region selection
    if SA_L >= 0.0:
        F = FL_s
    elif SM >= 0.0:
        F = FL_dd
    elif SA_R > 0.0:
        F = FR_dd
    else:
        F = FR_s

    # NaN safety
    if not np.all(np.isfinite(F)):
        S_max = max(abs(SL), abs(SR))
        F = 0.5*(FL + FR) - 0.5*S_max*(UR - UL)

    return F.astype(np.float32)


def hlld_flux_np(
    UL: np.ndarray,
    UR: np.ndarray,
    direction: int,
) -> np.ndarray:
    """NumPy reference HLLD flux over a 2-D interface grid.

    Parameters
    ----------
    UL, UR : ndarray, shape (10, n_ir, n_iz)
    direction : int, 0 = r-flux, 1 = z-flux

    Returns
    -------
    ndarray, shape (10, n_ir, n_iz), float32
    """
    assert UL.ndim == 3 and UL.shape[0] == 10
    assert UR.shape == UL.shape
    _, n_ir, n_iz = UL.shape
    flux = np.zeros_like(UL, dtype=np.float32)
    for i in range(n_ir):
        for j in range(n_iz):
            flux[:, i, j] = _hlld_1d(UL[:, i, j].astype(np.float64),
                                      UR[:, i, j].astype(np.float64),
                                      direction)
    return flux


# ============================================================================
# Kernel 3 reference – Cylindrical Geometric Source Terms
# ============================================================================

def cyl_source_np(
    prim: np.ndarray,
    r_arr: np.ndarray,
    dr: float,
) -> np.ndarray:
    """NumPy reference for cylindrical geometric source terms.

    Parameters
    ----------
    prim : ndarray, shape (10, nr, nz)
        Primitive state: [rho, vr, vz, vθ, p, S, Br, Bz, Bθ, Ee]
    r_arr : ndarray, shape (nr,)
        Cell-centre radii [m].
    dr : float
        Radial cell spacing [m].

    Returns
    -------
    ndarray, shape (10, nr, nz), float32
    """
    assert prim.ndim == 3 and prim.shape[0] == 10
    prim  = prim.astype(np.float64)
    r_arr = r_arr.astype(np.float64)

    _, nr, nz = prim.shape
    src = np.zeros_like(prim, dtype=np.float64)

    rho    = prim[0]
    vr     = prim[1]
    vtheta = prim[3]
    p      = prim[4]
    Br     = prim[6]
    Bz     = prim[7]
    Btheta = prim[8]
    B2     = Br**2 + Bz**2 + Btheta**2

    # Radii array broadcast to (nr, nz)
    r2d = r_arr[:, np.newaxis]  # (nr, 1)

    # ---- Regular cells (ir > 0) ----
    f_r_mom  = (p + 0.5*B2 - Btheta**2) / r2d + rho * vtheta**2 / r2d
    f_th_mom = -(rho * vr * vtheta - Br * Btheta) / r2d
    f_Btheta = -(vr * Btheta - Br * vtheta) / r2d

    src[1] = f_r_mom
    src[3] = f_th_mom
    src[8] = f_Btheta

    # ---- L'Hôpital at ir = 0 ----
    if nr > 1:
        # r-momentum at ir=0
        f0 = (p[0]  + 0.5*B2[0]  - Btheta[0]**2  + rho[0]  * vtheta[0]**2)
        f1 = (p[1]  + 0.5*B2[1]  - Btheta[1]**2  + rho[1]  * vtheta[1]**2)
        src[1, 0, :] = (f1 - f0) / dr

        # θ-momentum and Bθ vanish at axis (vr=0, Br=0 from reflecting BC)
        src[3, 0, :] = 0.0
        src[8, 0, :] = 0.0

    return src.astype(np.float32)
