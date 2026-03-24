"""Custom Metal kernels, Python wrappers, and NumPy reference implementations for
magnetohydrodynamics on a cylindrical grid using MLX.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:
    import mlx.core as mx
except Exception:  # pragma: no cover - MLX is only available on Apple Silicon
    mx = None

# Physical constants
MU0: float = 4.0 * np.pi * 1e-7
GAMMA: float = 5.0 / 3.0

# Component ordering (structure-of-arrays)
COMPONENTS: Dict[str, int] = {
    "rho": 0,
    "vr": 1,
    "vz": 2,
    "vtheta": 3,
    "p": 4,
    "Srho": 5,
    "Br": 6,
    "Bz": 7,
    "Btheta": 8,
    "Ee": 9,
}

# ---------------------------------------------------------------------------
# Metal kernel sources

GHOST_PAD_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant float MU0 = 4.0f * M_PI_F * 1e-7f;
constant float EPS = 1e-6f;
constant uint COMP_RHO = 0;
constant uint COMP_VR = 1;
constant uint COMP_VZ = 2;
constant uint COMP_VTH = 3;
constant uint COMP_P = 4;
constant uint COMP_S = 5;
constant uint COMP_BR = 6;
constant uint COMP_BZ = 7;
constant uint COMP_BTH = 8;
constant uint COMP_EE = 9;

inline uint idx(uint c, uint r, uint z, uint nr, uint nz) {
    return c * nr * nz + r * nz + z;
}

kernel void ghost_pad(
    const device float* state [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& nr [[buffer(2)]],
    constant uint& nz [[buffer(3)]],
    constant uint& ng [[buffer(4)]],
    constant float& dr [[buffer(5)]],
    constant float& r_min [[buffer(6)]],
    constant float& current_I [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint r_out = gid.x;
    uint z = gid.y;
    uint comp = gid.z;
    uint padded_nr = nr + 2 * ng;
    if (r_out >= padded_nr || z >= nz || comp >= 10) {
        return;
    }

    int r_in = int(r_out) - int(ng);
    float val = 0.0f;

    if (r_in >= 0 && r_in < int(nr)) {
        uint src = idx(comp, uint(r_in), z, nr, nz);
        val = state[src];
    } else if (r_in < 0) {
        uint src_r = uint(-r_in - 1);
        if (src_r >= nr) src_r = nr - 1;
        uint src = idx(comp, src_r, z, nr, nz);
        float src_val = state[src];
        if (comp == COMP_VR || comp == COMP_BR || comp == COMP_BTH) {
            val = 0.0f;
        } else if (comp == COMP_VTH) {
            val = -src_val;
        } else {
            val = src_val;
        }
    } else {
        uint src_r = nr - 1;
        uint src = idx(comp, src_r, z, nr, nz);
        float src_val = state[src];
        if (comp == COMP_VR || comp == COMP_BR) {
            val = 0.0f;
        } else if (comp == COMP_BTH) {
            float r = r_min + (float(r_in) + 0.5f) * dr;
            float denom = max(r, EPS);
            val = (MU0 * current_I) / (2.0f * M_PI_F * denom);
        } else {
            val = src_val;
        }
    }

    // Axial boundaries: reflect at z=0, outflow at z=nz-1
    if (z == 0) {
        uint mirror_z = (nz > 1) ? 1 : 0;
        uint src_r = (r_in >= 0 && r_in < int(nr))
                         ? uint(r_in)
                         : (r_in < 0 ? uint(-r_in - 1) : nr - 1);
        uint src = idx(comp, src_r, mirror_z, nr, nz);
        float src_val = state[src];
        if (comp == COMP_VZ || comp == COMP_BZ) {
            val = -src_val;
        }
    } else if (z == nz - 1) {
        uint src_r = (r_in >= 0 && r_in < int(nr))
                         ? uint(r_in)
                         : (r_in < 0 ? uint(-r_in - 1) : nr - 1);
        uint src = idx(comp, src_r, nz - 1, nr, nz);
        val = state[src];
    }

    uint out_idx = comp * padded_nr * nz + r_out * nz + z;
    out[out_idx] = val;
}
"""


HLLD_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant float GAMMA = 5.0f / 3.0f;
constant float EPS = 1e-6f;
constant uint COMP_RHO = 0;
constant uint COMP_VR = 1;
constant uint COMP_VZ = 2;
constant uint COMP_VTH = 3;
constant uint COMP_P = 4;
constant uint COMP_S = 5;
constant uint COMP_BR = 6;
constant uint COMP_BZ = 7;
constant uint COMP_BTH = 8;
constant uint COMP_EE = 9;

inline uint idx(uint c, uint r, uint z, uint nr, uint nz) {
    return c * nr * nz + r * nz + z;
}

struct Prim {
    float rho, vr, vz, vt, p, s, br, bz, bt, ee;
};

inline float comp(const Prim& p, uint i) {
    switch (i) {
        case COMP_RHO: return p.rho;
        case COMP_VR: return p.vr;
        case COMP_VZ: return p.vz;
        case COMP_VTH: return p.vt;
        case COMP_P: return p.p;
        case COMP_S: return p.s;
        case COMP_BR: return p.br;
        case COMP_BZ: return p.bz;
        case COMP_BTH: return p.bt;
        default: return p.ee;
    }
}

inline Prim load_state(const device float* arr, uint r, uint z, uint nr, uint nz) {
    Prim st;
    st.rho = arr[idx(COMP_RHO, r, z, nr, nz)];
    st.vr = arr[idx(COMP_VR, r, z, nr, nz)];
    st.vz = arr[idx(COMP_VZ, r, z, nr, nz)];
    st.vt = arr[idx(COMP_VTH, r, z, nr, nz)];
    st.p = arr[idx(COMP_P, r, z, nr, nz)];
    st.s = arr[idx(COMP_S, r, z, nr, nz)];
    st.br = arr[idx(COMP_BR, r, z, nr, nz)];
    st.bz = arr[idx(COMP_BZ, r, z, nr, nz)];
    st.bt = arr[idx(COMP_BTH, r, z, nr, nz)];
    st.ee = arr[idx(COMP_EE, r, z, nr, nz)];
    return st;
}

inline void store_flux(device float* out, uint r, uint z, uint nr, uint nz, float flux[10]) {
    for (uint c = 0; c < 10; ++c) {
        out[idx(c, r, z, nr, nz)] = flux[c];
    }
}

inline float fast_speed(const Prim st) {
    float a2 = GAMMA * st.p / st.rho;
    float bt2 = st.bz * st.bz + st.bt * st.bt;
    float va2 = (st.br * st.br + bt2) / st.rho;
    float disc = (a2 - va2) * (a2 - va2) + 4.0f * a2 * bt2 / st.rho;
    disc = max(disc, 0.0f);
    float root = sqrt(disc);
    float cf2 = 0.5f * (a2 + va2 + root);
    cf2 = max(cf2, 0.0f);
    return sqrt(cf2);
}

inline void flux(const Prim st, float flux_out[10], float Bn) {
    float b2 = st.br * st.br + st.bz * st.bz + st.bt * st.bt;
    float v2 = st.vr * st.vr + st.vz * st.vz + st.vt * st.vt;
    float E = st.p / (GAMMA - 1.0f) + 0.5f * st.rho * v2 + 0.5f * b2;
    float pt = st.p + 0.5f * b2;

    flux_out[0] = st.rho * st.vr;
    flux_out[1] = st.rho * st.vr * st.vr + pt - st.br * Bn;
    flux_out[2] = st.rho * st.vr * st.vz - st.br * st.bz;
    flux_out[3] = st.rho * st.vr * st.vt - st.br * st.bt;
    flux_out[4] = (E + pt) * st.vr - Bn * (st.vr * st.br + st.vz * st.bz + st.vt * st.bt);
    flux_out[5] = st.s * st.vr;
    flux_out[6] = 0.0f;
    flux_out[7] = st.vr * st.bz - st.vz * Bn;
    flux_out[8] = st.vr * st.bt - st.vt * Bn;
    flux_out[9] = st.ee * st.vr;
}

kernel void hlld_flux(
    const device float* left [[buffer(0)]],
    const device float* right [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& nr [[buffer(3)]],
    constant uint& nz [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint r = gid.x;
    uint z = gid.y;
    if (r >= nr || z >= nz) return;

    Prim L = load_state(left, r, z, nr, nz);
    Prim R = load_state(right, r, z, nr, nz);

    float cfL = fast_speed(L);
    float cfR = fast_speed(R);
    float SL = min(L.vr - cfL, R.vr - cfR);
    float SR = max(L.vr + cfL, R.vr + cfR);
    float Bn = 0.5f * (L.br + R.br);

    float FL[10];
    float FR[10];
    flux(L, FL, Bn);
    flux(R, FR, Bn);

    // Contact speed
    float ptL = L.p + 0.5f * (L.br * L.br + L.bz * L.bz + L.bt * L.bt);
    float ptR = R.p + 0.5f * (R.br * R.br + R.bz * R.bz + R.bt * R.bt);
    float SM_num = (SR - R.vr) * R.rho * R.vr - (SL - L.vr) * L.rho * L.vr + ptL - ptR;
    float SM_den = (SR - R.vr) * R.rho - (SL - L.vr) * L.rho;
    if (fabs(SM_den) < EPS) SM_den = (SM_den >= 0.0f ? EPS : -EPS);
    float SM = SM_num / SM_den;

    float denomL = SL - SM;
    float denomR = SR - SM;
    denomL = (fabs(denomL) < EPS) ? (denomL >= 0.0f ? EPS : -EPS) : denomL;
    denomR = (fabs(denomR) < EPS) ? (denomR >= 0.0f ? EPS : -EPS) : denomR;

    float rhoL_star = L.rho * (SL - L.vr) / denomL;
    float rhoR_star = R.rho * (SR - R.vr) / denomR;

    float qL = (SL - L.vr) / denomL;
    float qR = (SR - R.vr) / denomR;

    float bzL_star = L.bz * qL;
    float btL_star = L.bt * qL;
    float bzR_star = R.bz * qR;
    float btR_star = R.bt * qR;

    float vzL_star = L.vz - (Bn * (L.bz - bzL_star)) / (L.rho * (SL - L.vr));
    float vtL_star = L.vt - (Bn * (L.bt - btL_star)) / (L.rho * (SL - L.vr));
    float vzR_star = R.vz - (Bn * (R.bz - bzR_star)) / (R.rho * (SR - R.vr));
    float vtR_star = R.vt - (Bn * (R.bt - btR_star)) / (R.rho * (SR - R.vr));

    float v2L = L.vr * L.vr + L.vz * L.vz + L.vt * L.vt;
    float b2L = L.br * L.br + L.bz * L.bz + L.bt * L.bt;
    float EL = L.p / (GAMMA - 1.0f) + 0.5f * L.rho * v2L + 0.5f * b2L;

    float v2R = R.vr * R.vr + R.vz * R.vz + R.vt * R.vt;
    float b2R = R.br * R.br + R.bz * R.bz + R.bt * R.bt;
    float ER = R.p / (GAMMA - 1.0f) + 0.5f * R.rho * v2R + 0.5f * b2R;

    float EL_star = ( (SL - L.vr) * EL - L.p * L.vr + (ptL + L.br * (SM - L.vr)) * SM - Bn * (L.vr * L.br + L.vz * L.bz + L.vt * L.bt - SM * L.br - vzL_star * bzL_star - vtL_star * btL_star) ) / (SL - SM);
    float ER_star = ( (SR - R.vr) * ER - R.p * R.vr + (ptR + R.br * (SM - R.vr)) * SM - Bn * (R.vr * R.br + R.vz * R.bz + R.vt * R.bt - SM * R.br - vzR_star * bzR_star - vtR_star * btR_star) ) / (SR - SM);

    float UL_star[10] = {
        rhoL_star,
        rhoL_star * SM,
        rhoL_star * vzL_star,
        rhoL_star * vtL_star,
        EL_star,
        L.s * qL,
        Bn,
        bzL_star,
        btL_star,
        L.ee * qL
    };

    float UR_star[10] = {
        rhoR_star,
        rhoR_star * SM,
        rhoR_star * vzR_star,
        rhoR_star * vtR_star,
        ER_star,
        R.s * qR,
        Bn,
        bzR_star,
        btR_star,
        R.ee * qR
    };

    float FL_star[10];
    float FR_star[10];
    for (uint i = 0; i < 10; ++i) {
        FL_star[i] = FL[i] + SL * (UL_star[i] - (i == COMP_BR ? L.br : comp(L, i)));
        FR_star[i] = FR[i] + SR * (UR_star[i] - (i == COMP_BR ? R.br : comp(R, i)));
    }

    float flux_out[10];
    bool bad = (isnan(SM) || isnan(SL) || isnan(SR));
    if (!bad) {
        for (uint i = 0; i < 10; ++i) {
            bad = bad || isnan(FL[i]) || isnan(FR[i]) || isnan(FL_star[i]) || isnan(FR_star[i]);
        }
    }

    if (bad) {
        float alpha = max(fabs(L.vr) + cfL, fabs(R.vr) + cfR);
        for (uint i = 0; i < 10; ++i) {
            float UL_i = (i == COMP_BR ? L.br : comp(L, i));
            float UR_i = (i == COMP_BR ? R.br : comp(R, i));
            flux_out[i] = 0.5f * (FL[i] + FR[i]) - 0.5f * alpha * (UR_i - UL_i);
        }
    } else if (0.0f <= SL) {
        for (uint i = 0; i < 10; ++i) flux_out[i] = FL[i];
    } else if (SL <= 0.0f && 0.0f <= SM) {
        for (uint i = 0; i < 10; ++i) flux_out[i] = FL_star[i];
    } else if (SM <= 0.0f && 0.0f <= SR) {
        for (uint i = 0; i < 10; ++i) flux_out[i] = FR_star[i];
    } else {
        for (uint i = 0; i < 10; ++i) flux_out[i] = FR[i];
    }

    store_flux(out, r, z, nr, nz, flux_out);
}
"""


GEOMETRIC_SOURCES_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant float SMALL_R = 1e-6f;
constant uint COMP_RHO = 0;
constant uint COMP_VR = 1;
constant uint COMP_VZ = 2;
constant uint COMP_VTH = 3;
constant uint COMP_P = 4;
constant uint COMP_S = 5;
constant uint COMP_BR = 6;
constant uint COMP_BZ = 7;
constant uint COMP_BTH = 8;
constant uint COMP_EE = 9;

inline uint idx(uint c, uint r, uint z, uint nr, uint nz) {
    return c * nr * nz + r * nz + z;
}

kernel void geometric_sources(
    const device float* prim [[buffer(0)]],
    const device float* radii [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& nr [[buffer(3)]],
    constant uint& nz [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint r = gid.x;
    uint z = gid.y;
    if (r >= nr || z >= nz) return;

    float rho = prim[idx(COMP_RHO, r, z, nr, nz)];
    float vr = prim[idx(COMP_VR, r, z, nr, nz)];
    float vz = prim[idx(COMP_VZ, r, z, nr, nz)];
    float vth = prim[idx(COMP_VTH, r, z, nr, nz)];
    float p = prim[idx(COMP_P, r, z, nr, nz)];
    float Br = prim[idx(COMP_BR, r, z, nr, nz)];
    float Bz = prim[idx(COMP_BZ, r, z, nr, nz)];
    float Bth = prim[idx(COMP_BTH, r, z, nr, nz)];

    float r_c = radii[r];
    float r_safe = max(r_c, SMALL_R);

    float B2 = Br * Br + Bz * Bz + Bth * Bth;
    float source[10];
    for (uint i = 0; i < 10; ++i) source[i] = 0.0f;

    float pressure_term;
    if (r_c <= SMALL_R && r + 1 < nr) {
        // L'Hopital: use dp/dr instead of p/r
        float dp = prim[idx(COMP_P, r + 1, z, nr, nz)] - p;
        float dr = radii[r + 1] - r_c;
        pressure_term = dp / max(dr, SMALL_R);
    } else {
        pressure_term = p / r_safe;
    }

    source[COMP_VR] = (pressure_term + 0.5f * B2 / r_safe - (Bth * Bth) / r_safe) + rho * vth * vth / r_safe;
    source[COMP_VTH] = -((rho * vr * vth - Br * Bth) / r_safe);
    source[COMP_BTH] = -((vr * Bth - Br * vth) / r_safe);

    for (uint c = 0; c < 10; ++c) {
        out[idx(c, r, z, nr, nz)] = source[c];
    }
}
"""


# ---------------------------------------------------------------------------
# Python helper utilities

def _require_mx() -> None:
    if mx is None:
        raise ImportError(
            "mlx.core is required for GPU kernels. Install MLX on Apple Silicon."
        )


def _to_mx_float32(arr: "mx.array") -> "mx.array":
    return mx.array(arr, dtype=mx.float32)


def build_ghost_padding_kernel():
    """Return compiled ghost padding Metal kernel."""
    _require_mx()
    return mx.fast.metal_kernel(GHOST_PAD_MSL, "ghost_pad")


def build_hlld_kernel():
    """Return compiled HLLD Metal kernel."""
    _require_mx()
    return mx.fast.metal_kernel(HLLD_MSL, "hlld_flux")


def build_geometric_source_kernel():
    """Return compiled geometric source Metal kernel."""
    _require_mx()
    return mx.fast.metal_kernel(GEOMETRIC_SOURCES_MSL, "geometric_sources")


def ghost_pad(
    state: "mx.array",
    current_I: float,
    dr: float,
    r_min: Optional[float] = None,
    ng: int = 3,
) -> "mx.array":
    """Apply ghost-cell padding with electrode boundary conditions.

    Parameters
    ----------
    state : mx.array
        Primitive state array with shape (10, nr, nz).
    current_I : float
        Electrode current in Amps.
    dr : float
        Radial spacing.
    r_min : float, optional
        Radius of first cell center. Defaults to dr / 2.
    ng : int
        Number of radial ghost cells.
    """

    _require_mx()
    state = _to_mx_float32(state)
    nr, nz = int(state.shape[1]), int(state.shape[2])
    r_min = dr * 0.5 if r_min is None else r_min
    out = mx.empty((10, nr + 2 * ng, nz), dtype=state.dtype, device=state.device)
    kernel = build_ghost_padding_kernel()
    grid = (nr + 2 * ng, nz, 10)
    # mx.fast.metal_kernel expects inputs as a list and the output as a separate argument.
    kernel([state], out, constants=[nr, nz, ng, dr, r_min, float(current_I)], grid=grid)
    return out


def _hlld_flux_single(
    UL: np.ndarray, UR: np.ndarray
) -> np.ndarray:
    """Reference HLLD-like flux for a single interface (NumPy)."""
    rhoL, vrL, vzL, vtL, pL, sL, brL, bzL, btL, eeL = UL
    rhoR, vrR, vzR, vtR, pR, sR, brR, bzR, btR, eeR = UR

    def fast_speed(rho, p, br, bz, bt, vr):
        a2 = GAMMA * p / rho
        bt2 = bz * bz + bt * bt
        va2 = (br * br + bt2) / rho
        disc = (a2 - va2) ** 2 + 4.0 * a2 * bt2 / rho
        disc = max(disc, 0.0)
        cf2 = 0.5 * (a2 + va2 + np.sqrt(disc))
        return np.sqrt(max(cf2, 0.0))

    cfL = fast_speed(rhoL, pL, brL, bzL, btL, vrL)
    cfR = fast_speed(rhoR, pR, brR, bzR, btR, vrR)
    SL = min(vrL - cfL, vrR - cfR)
    SR = max(vrL + cfL, vrR + cfR)
    Bn = 0.5 * (brL + brR)

    def flux(state):
        rho, vr, vz, vt, p, s, br, bz, bt, ee = state
        b2 = br * br + bz * bz + bt * bt
        v2 = vr * vr + vz * vz + vt * vt
        E = p / (GAMMA - 1.0) + 0.5 * rho * v2 + 0.5 * b2
        pt = p + 0.5 * b2
        return np.array(
            [
                rho * vr,
                rho * vr * vr + pt - br * Bn,
                rho * vr * vz - br * bz,
                rho * vr * vt - br * bt,
                (E + pt) * vr - Bn * (vr * br + vz * bz + vt * bt),
                s * vr,
                0.0,
                vr * bz - vz * Bn,
                vr * bt - vt * Bn,
                ee * vr,
            ],
            dtype=np.float32,
        )

    FL = flux(UL)
    FR = flux(UR)

    ptL = pL + 0.5 * (brL * brL + bzL * bzL + btL * btL)
    ptR = pR + 0.5 * (brR * brR + bzR * bzR + btR * btR)
    SM = ((SR - vrR) * rhoR * vrR - (SL - vrL) * rhoL * vrL + ptL - ptR) / (
        (SR - vrR) * rhoR - (SL - vrL) * rhoL
    )

    rhoL_star = rhoL * (SL - vrL) / (SL - SM)
    rhoR_star = rhoR * (SR - vrR) / (SR - SM)
    qL = (SL - vrL) / (SL - SM)
    qR = (SR - vrR) / (SR - SM)

    bzL_star = bzL * qL
    btL_star = btL * qL
    bzR_star = bzR * qR
    btR_star = btR * qR

    vzL_star = vzL - (Bn * (bzL - bzL_star)) / (rhoL * (SL - vrL))
    vtL_star = vtL - (Bn * (btL - btL_star)) / (rhoL * (SL - vrL))
    vzR_star = vzR - (Bn * (bzR - bzR_star)) / (rhoR * (SR - vrR))
    vtR_star = vtR - (Bn * (btR - btR_star)) / (rhoR * (SR - vrR))

    def energy(rho, vr, vz, vt, p, br, bz, bt):
        v2 = vr * vr + vz * vz + vt * vt
        b2 = br * br + bz * bz + bt * bt
        return p / (GAMMA - 1.0) + 0.5 * rho * v2 + 0.5 * b2

    EL = energy(rhoL, vrL, vzL, vtL, pL, brL, bzL, btL)
    ER = energy(rhoR, vrR, vzR, vtR, pR, brR, bzR, btR)

    EL_star = (
        (SL - vrL) * EL
        - pL * vrL
        + (ptL + brL * (SM - vrL)) * SM
        - Bn
        * (vrL * brL + vzL * bzL + vtL * btL - SM * brL - vzL_star * bzL_star - vtL_star * btL_star)
    ) / (SL - SM)

    ER_star = (
        (SR - vrR) * ER
        - pR * vrR
        + (ptR + brR * (SM - vrR)) * SM
        - Bn
        * (vrR * brR + vzR * bzR + vtR * btR - SM * brR - vzR_star * bzR_star - vtR_star * btR_star)
    ) / (SR - SM)

    UL_star = np.array(
        [
            rhoL_star,
            rhoL_star * SM,
            rhoL_star * vzL_star,
            rhoL_star * vtL_star,
            EL_star,
            sL * qL,
            Bn,
            bzL_star,
            btL_star,
            eeL * qL,
        ],
        dtype=np.float32,
    )
    UR_star = np.array(
        [
            rhoR_star,
            rhoR_star * SM,
            rhoR_star * vzR_star,
            rhoR_star * vtR_star,
            ER_star,
            sR * qR,
            Bn,
            bzR_star,
            btR_star,
            eeR * qR,
        ],
        dtype=np.float32,
    )

    FL_star = FL + SL * (UL_star - UL)
    FR_star = FR + SR * (UR_star - UR)

    bad = (
        np.isnan([SL, SR, SM]).any()
        or np.isnan(FL).any()
        or np.isnan(FR).any()
        or np.isnan(FL_star).any()
        or np.isnan(FR_star).any()
    )
    if bad:
        alpha = max(abs(vrL) + cfL, abs(vrR) + cfR)
        return 0.5 * (FL + FR) - 0.5 * alpha * (UR - UL)

    if 0.0 <= SL:
        return FL
    if SL <= 0.0 <= SM:
        return FL_star
    if SM <= 0.0 <= SR:
        return FR_star
    return FR


def hlld_flux_numpy(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Vectorized NumPy reference for the HLLD flux.

    Parameters
    ----------
    left, right : np.ndarray
        Arrays shaped (10, nr, nz) with primitive variables.
    """

    assert left.shape == right.shape
    comps, nr, nz = left.shape
    out = np.zeros((10, nr, nz), dtype=np.float32)
    for i in range(nr):
        for k in range(nz):
            out[:, i, k] = _hlld_flux_single(
                left[:, i, k].astype(np.float32), right[:, i, k].astype(np.float32)
            )
    return out


def hlld_flux(
    left: "mx.array", right: "mx.array"
) -> "mx.array":
    """Compute fluxes using the Metal HLLD kernel."""
    _require_mx()
    left = _to_mx_float32(left)
    right = _to_mx_float32(right)
    nr, nz = int(left.shape[1]), int(left.shape[2])
    out = mx.empty((10, nr, nz), dtype=mx.float32, device=left.device)
    kernel = build_hlld_kernel()
    kernel([left, right], out, constants=[nr, nz], grid=(nr, nz))
    return out


def ghost_pad_numpy(
    state: np.ndarray, current_I: float, dr: float, r_min: Optional[float] = None, ng: int = 3
) -> np.ndarray:
    """NumPy reference for radial ghost padding."""
    comps, nr, nz = state.shape
    assert comps == 10
    r_min = dr * 0.5 if r_min is None else r_min
    out = np.zeros((10, nr + 2 * ng, nz), dtype=np.float32)

    def clamp_r(r_in: int) -> int:
        if r_in >= 0:
            return min(max(r_in, 0), nr - 1)
        return min(-r_in - 1, nr - 1)

    for r_out in range(nr + 2 * ng):
        r_in = r_out - ng
        for z in range(nz):
            for c in range(10):
                if 0 <= r_in < nr:
                    val = state[c, r_in, z]
                elif r_in < 0:
                    src_r = clamp_r(r_in)
                    src = state[c, src_r, z]
                    if c in (COMPONENTS["vr"], COMPONENTS["Br"], COMPONENTS["Btheta"]):
                        val = 0.0
                    elif c == COMPONENTS["vtheta"]:
                        val = -src
                    else:
                        val = src
                else:
                    src = state[c, nr - 1, z]
                    if c in (COMPONENTS["vr"], COMPONENTS["Br"]):
                        val = 0.0
                    elif c == COMPONENTS["Btheta"]:
                        r_val = r_min + (r_in + 0.5) * dr
                        val = (MU0 * current_I) / (2.0 * np.pi * max(r_val, 1e-6))
                    else:
                        val = src

                # axial
                if z == 0 and c in (COMPONENTS["vz"], COMPONENTS["Bz"]):
                    mirror_z = 1 if nz > 1 else 0
                    src_r = clamp_r(r_in)
                    val = -state[c, src_r, mirror_z]
                elif z == nz - 1:
                    src_r = clamp_r(r_in)
                    val = state[c, src_r, nz - 1]

                out[c, r_out, z] = val
    return out


def cylindrical_sources_numpy(prim: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Reference geometric source term."""
    comps, nr, nz = prim.shape
    out = np.zeros_like(prim, dtype=np.float32)
    eps = 1e-6
    for i in range(nr):
        r = radii[i]
        r_safe = max(r, eps)
        for k in range(nz):
            rho, vr, vz, vt, p, _, Br, Bz, Bth, _ = prim[:, i, k]
            B2 = Br * Br + Bz * Bz + Bth * Bth
            if r <= eps and i + 1 < nr:
                dp = prim[COMPONENTS["p"], i + 1, k] - p
                dr = radii[i + 1] - r
                pressure_term = dp / max(dr, eps)
            else:
                pressure_term = p / r_safe
            out[COMPONENTS["vr"], i, k] = (
                pressure_term + 0.5 * B2 / r_safe - (Bth * Bth) / r_safe + rho * vt * vt / r_safe
            )
            out[COMPONENTS["vtheta"], i, k] = -((rho * vr * vt - Br * Bth) / r_safe)
            out[COMPONENTS["Btheta"], i, k] = -((vr * Bth - Br * vt) / r_safe)
    return out


def cylindrical_sources(
    prim: "mx.array", radii: "mx.array"
) -> "mx.array":
    """Compute cylindrical geometric source terms."""
    _require_mx()
    prim = _to_mx_float32(prim)
    radii = _to_mx_float32(radii)
    nr, nz = int(prim.shape[1]), int(prim.shape[2])
    out = mx.zeros_like(prim)
    kernel = build_geometric_source_kernel()
    kernel([prim, radii], out, constants=[nr, nz], grid=(nr, nz))
    return out


def recommended_thread_group(nr: int, nz: int) -> Tuple[int, int, int]:
    """Return a good default threadgroup size for M3 Pro (14-core GPU).

    Parameters
    ----------
    nr, nz : int
        Problem dimensions along radial and axial directions. The values are
        used to clamp the suggested radial/axial sizes so they never exceed
        the grid.

    Returns
    -------
        (tg_r, tg_z, tg_comp) : tuple[int, int, int]
        Threadgroup extents for radial, axial, and component dimensions. The
        third entry is fixed at 4 for the ghost-padding kernel's 3-D grid,
        while the first two entries are clamped to the provided sizes. For
        2-D kernels (HLLD flux or geometric sources) use only ``tg_r`` and
        ``tg_z``.
    """
    # Favor ~256 threads per group while keeping 3D occupancy reasonable.
    tg_r = 16 if nr >= 16 else max(1, nr)
    tg_z = 8 if nz >= 8 else max(1, nz)
    tg_comp = 4
    return tg_r, tg_z, tg_comp
