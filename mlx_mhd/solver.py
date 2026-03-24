"""High-level cylindrical MHD solver utilities built on the kernel package.

The implementation focuses on a robust dual-energy finite-volume scaffold:

* conserved/primitive conversion with entropy-based pressure recovery
* component-wise WENO5-Z reconstruction
* HLLD flux assembly in radial and axial directions
* SSP-RK3 stepping with entropy re-synchronisation
* cylindrical geometric source integration
* lightweight CT-style magnetic update and implicit resistive diffusion
* circuit coupling helpers for PF-1000-style DPF simulations

The numerics are intentionally conservative and NumPy-first so they can be
tested on non-Apple CI hosts while still interoperating with the MLX kernels
exported elsewhere in the package.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Optional

import numpy as np

from .kernels import COMPONENTS, GAMMA, MU0, cylindrical_sources_numpy, hlld_flux_numpy

RHO = COMPONENTS["rho"]
MR = COMPONENTS["rho_vr"]
MZ = COMPONENTS["rho_vz"]
MT = COMPONENTS["rho_vtheta"]
ENERGY = COMPONENTS["E"]
SRHO = COMPONENTS["Srho"]
BR = COMPONENTS["Br"]
BZ = COMPONENTS["Bz"]
BTH = COMPONENTS["Btheta"]
EE = COMPONENTS["Ee"]

K_B = 1.380649e-23
M_DEUTERIUM = 3.343583719e-27
EV_TO_JOULES = 1.602176634e-19


@dataclass(frozen=True)
class SolverConfig:
    """Numerical and physical controls for the solver."""

    gamma: float = GAMMA
    density_floor: float = 1e-8
    pressure_floor: float = 1e-20
    electron_energy_floor: float = 1e-20
    eta_switch_low: float = 1e-5
    eta_switch_high: float = 1e-2
    cfl: float = 0.35
    resistivity: float = 0.0
    bremsstrahlung: bool = True
    enable_ct: bool = True
    enable_resistive_diffusion: bool = True
    entropy_sync_compression: float = 0.33
    entropy_sync_pressure_jump: float = 0.33
    entropy_sync_eta: float = 0.01
    use_spitzer_resistivity: bool = False
    spitzer_Z: float = 1.0
    spitzer_lnA: float = 10.0
    spitzer_eta_floor: float = 1e-8
    spitzer_eta_cap: float = 1e-2


@dataclass(frozen=True)
class CylindricalGrid:
    """Uniform axisymmetric cylindrical mesh with cell-centred state storage."""

    nr: int
    nz: int
    dr: float
    dz: float
    r_min: Optional[float] = None
    z_min: float = 0.0
    ng: int = 3

    def __post_init__(self) -> None:
        if self.nr < 4 or self.nz < 4:
            raise ValueError("grid must have at least 4x4 cells")
        if self.dr <= 0.0 or self.dz <= 0.0:
            raise ValueError("dr and dz must be positive")

    @property
    def inner_radius(self) -> float:
        return self.dr * 0.5 if self.r_min is None else self.r_min

    @property
    def radii(self) -> np.ndarray:
        return self.inner_radius + np.arange(self.nr, dtype=np.float64) * self.dr

    @property
    def axial(self) -> np.ndarray:
        return self.z_min + (np.arange(self.nz, dtype=np.float64) + 0.5) * self.dz

    @property
    def radial_faces(self) -> np.ndarray:
        start = self.inner_radius - 0.5 * self.dr
        faces = start + np.arange(self.nr + 1, dtype=np.float64) * self.dr
        return np.maximum(faces, 0.0)

    @property
    def axial_faces(self) -> np.ndarray:
        return self.z_min + np.arange(self.nz + 1, dtype=np.float64) * self.dz

    @property
    def radial_face_areas(self) -> np.ndarray:
        return 2.0 * math.pi * self.radial_faces * self.dz

    @property
    def axial_face_areas(self) -> np.ndarray:
        rf = self.radial_faces
        return math.pi * (rf[1:] ** 2 - rf[:-1] ** 2)

    @property
    def cell_volumes(self) -> np.ndarray:
        return self.axial_face_areas[:, None] * self.dz


@dataclass(frozen=True)
class CircuitParameters:
    """External RLC circuit parameters."""

    V0: float
    C: float
    L0: float
    R0: float
    anode_radius: float
    cathode_radius: float


@dataclass(frozen=True)
class CircuitState:
    """Dynamic circuit state advanced alongside the plasma."""

    current: float = 0.0
    capacitor_voltage: float = 0.0
    plasma_inductance: float = 0.0
    time: float = 0.0
    back_emf: float = 0.0


PF1000_CIRCUIT = CircuitParameters(
    V0=27_000.0,
    C=1.332e-3,
    L0=33.5e-9,
    R0=2.3e-3,
    anode_radius=0.0115,
    cathode_radius=0.16,
)


@dataclass(frozen=True)
class ValidationTarget:
    """A single validation metric with its acceptable reference range."""

    label: str
    unit: str
    reference_low: float
    reference_high: float


PF1000_VALIDATION_TARGETS: dict[str, "ValidationTarget"] = {
    "peak_current": ValidationTarget(
        "Peak discharge current", "A", 1.0e6, 2.5e6,
    ),
    "time_of_peak_current": ValidationTarget(
        "Time to peak current", "s", 4.0e-6, 8.0e-6,
    ),
    "current_dip_fraction": ValidationTarget(
        "Current dip at pinch", "", 0.02, 0.30,
    ),
    "pinch_time": ValidationTarget(
        "Pinch time", "s", 4.0e-6, 10.0e-6,
    ),
    "peak_dI_dt": ValidationTarget(
        "Peak |dI/dt|", "A/s", 1.0e11, 1.0e13,
    ),
    "inductance_at_pinch": ValidationTarget(
        "Inductance at pinch", "H", 10.0e-9, 60.0e-9,
    ),
    "mean_sheath_speed": ValidationTarget(
        "Mean axial sheath speed", "m/s", 2.0e4, 3.0e5,
    ),
    "radiated_energy_fraction": ValidationTarget(
        "Radiated energy fraction", "", 0.0, 0.30,
    ),
}


@dataclass
class PF1000RunResult:
    """Data returned by :func:`run_pf1000_simulation`."""

    times: np.ndarray
    currents: np.ndarray
    voltages: np.ndarray
    inductances: np.ndarray
    state: np.ndarray
    dI_dt: np.ndarray
    sheath_positions: np.ndarray
    radiated_energy: np.ndarray
    stored_energy: float = 0.0

    @property
    def peak_current(self) -> float:
        """Maximum absolute discharge current in amperes."""
        return float(np.max(np.abs(self.currents)))

    @property
    def time_of_peak_current(self) -> float:
        return float(self.times[int(np.argmax(np.abs(self.currents)))])

    @property
    def pinch_time(self) -> float:
        """Time of maximum |dI/dt|, indicating the pinch phase."""
        return float(self.times[int(np.argmax(np.abs(self.dI_dt)))])

    @property
    def current_dip_fraction(self) -> float:
        """Fractional current dip after peak, indicating energy transfer to the pinch."""
        idx_peak = int(np.argmax(np.abs(self.currents)))
        i_peak = np.abs(self.currents[idx_peak])
        if idx_peak >= len(self.currents) - 1 or i_peak < 1e-6:
            return 0.0
        i_after = np.min(np.abs(self.currents[idx_peak:]))
        return float((i_peak - i_after) / i_peak)

    @property
    def peak_dI_dt(self) -> float:
        return float(np.max(np.abs(self.dI_dt)))

    @property
    def inductance_at_pinch(self) -> float:
        idx = int(np.argmax(np.abs(self.dI_dt)))
        return float(self.inductances[idx])

    @property
    def mean_sheath_speed(self) -> float:
        """Average axial sheath velocity from position trace."""
        if len(self.sheath_positions) < 2:
            return 0.0
        dz = np.abs(np.diff(self.sheath_positions))
        dt = np.diff(self.times)
        valid = dt > 0.0
        if not np.any(valid):
            return 0.0
        return float(np.mean(dz[valid] / dt[valid]))

    @property
    def radiated_energy_fraction(self) -> float:
        if self.stored_energy <= 0.0:
            return 0.0
        return float(self.radiated_energy[-1] / self.stored_energy)


def _ensure_state_shape(state: np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != 10:
        raise ValueError("state must have shape (10, nr, nz)")
    return arr


def _smoothstep(x: np.ndarray) -> np.ndarray:
    y = np.clip(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)


def _electron_temperature_eV(
    state: np.ndarray,
    ne: np.ndarray,
    config: SolverConfig,
) -> np.ndarray:
    """Compute electron temperature in eV from the conserved state and number density."""
    return (
        (2.0 / 3.0)
        * np.maximum(state[EE].astype(np.float64), config.electron_energy_floor)
        / np.maximum(ne * K_B, 1e-30)
        * K_B / EV_TO_JOULES
    )


def spitzer_resistivity(
    Te_eV: np.ndarray,
    *,
    Z: float = 1.0,
    lnA: float = 10.0,
    eta_floor: float = 1e-8,
    eta_cap: float = 1e-2,
) -> np.ndarray:
    """Classical Spitzer resistivity in Ω·m.

    Uses the simplified form ``η = 5.2×10⁻⁵ Z ln(Λ) / T_e^{3/2}`` where
    ``T_e`` is the electron temperature in eV.
    """
    Te = np.maximum(np.asarray(Te_eV, dtype=np.float64), 0.1)
    eta = 5.2e-5 * Z * lnA / Te**1.5
    return np.clip(eta, eta_floor, eta_cap)


def extract_sheath_position(state: np.ndarray, grid: "CylindricalGrid") -> float:
    """Return the axial position of the density-weighted current sheath."""
    state = _ensure_state_shape(state).astype(np.float64)
    rho = np.maximum(state[RHO], 0.0)
    r = grid.radii[:, None]
    profile = np.sum(rho * r * grid.dr, axis=0)
    return float(grid.axial[int(np.argmax(profile))])


def _estimate_radiated_power(
    state: np.ndarray,
    grid: "CylindricalGrid",
    config: SolverConfig,
) -> float:
    """Volume-integrated bremsstrahlung power in watts."""
    if not config.bremsstrahlung:
        return 0.0
    state = _ensure_state_shape(state).astype(np.float64)
    rho = np.maximum(state[RHO], config.density_floor)
    ne = rho / M_DEUTERIUM
    te = (
        (2.0 / 3.0)
        * np.maximum(state[EE], config.electron_energy_floor)
        / np.maximum(ne * K_B, 1e-30)
    )
    q_brem = 1.42e-40 * ne**2 * np.sqrt(np.maximum(te, 0.0))
    return float(np.sum(q_brem * grid.cell_volumes))


def recover_pressure(
    state: np.ndarray,
    *,
    gamma: float = GAMMA,
    eta1: float = 1e-5,
    eta2: float = 1e-2,
    density_floor: float = 1e-8,
    pressure_floor: float = 1e-20,
) -> np.ndarray:
    """Recover pressure using a dual-energy entropy switch."""

    state = _ensure_state_shape(state).astype(np.float64)
    rho = np.maximum(state[RHO], density_floor)
    ke = 0.5 * (state[MR] ** 2 + state[MZ] ** 2 + state[MT] ** 2) / rho
    me = 0.5 * (state[BR] ** 2 + state[BZ] ** 2 + state[BTH] ** 2)
    p_s = np.maximum(state[SRHO] * rho ** (gamma - 1.0), pressure_floor)
    p_e = (gamma - 1.0) * (state[ENERGY] - ke - me)
    eta = p_s / np.maximum(np.abs(state[ENERGY]), 1e-30)
    w = _smoothstep((eta - eta1) / (eta2 - eta1))
    return np.maximum(w * p_e + (1.0 - w) * p_s, pressure_floor).astype(np.float32)


def primitive_to_conserved(
    primitive: np.ndarray,
    *,
    gamma: float = GAMMA,
    density_floor: float = 1e-8,
    pressure_floor: float = 1e-20,
) -> np.ndarray:
    """Convert primitive state ``[rho, vr, vz, vtheta, p, S, Br, Bz, Btheta, Ee]``."""

    primitive = _ensure_state_shape(primitive).astype(np.float64)
    rho = np.maximum(primitive[RHO], density_floor)
    vr = primitive[MR]
    vz = primitive[MZ]
    vt = primitive[MT]
    p = np.maximum(primitive[ENERGY], pressure_floor)
    br = primitive[BR]
    bz = primitive[BZ]
    bth = primitive[BTH]
    ee = np.maximum(primitive[EE], pressure_floor)
    v2 = vr**2 + vz**2 + vt**2
    b2 = br**2 + bz**2 + bth**2
    conserved = np.zeros_like(primitive, dtype=np.float64)
    conserved[RHO] = rho
    conserved[MR] = rho * vr
    conserved[MZ] = rho * vz
    conserved[MT] = rho * vt
    conserved[ENERGY] = p / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * b2
    conserved[SRHO] = p / np.maximum(rho ** (gamma - 1.0), pressure_floor)
    conserved[BR] = br
    conserved[BZ] = bz
    conserved[BTH] = bth
    conserved[EE] = ee
    return conserved.astype(np.float32)


def conserved_to_primitive(
    state: np.ndarray,
    *,
    config: SolverConfig = SolverConfig(),
) -> np.ndarray:
    """Convert conserved state to primitive variables using dual-energy pressure."""

    state = _ensure_state_shape(state).astype(np.float64)
    rho = np.maximum(state[RHO], config.density_floor)
    primitive = np.zeros_like(state, dtype=np.float64)
    primitive[RHO] = rho
    primitive[MR] = state[MR] / rho
    primitive[MZ] = state[MZ] / rho
    primitive[MT] = state[MT] / rho
    pressure = recover_pressure(
        state,
        gamma=config.gamma,
        eta1=config.eta_switch_low,
        eta2=config.eta_switch_high,
        density_floor=config.density_floor,
        pressure_floor=config.pressure_floor,
    ).astype(np.float64)
    primitive[ENERGY] = pressure
    primitive[SRHO] = pressure / np.maximum(rho**config.gamma, config.pressure_floor)
    primitive[BR] = state[BR]
    primitive[BZ] = state[BZ]
    primitive[BTH] = state[BTH]
    primitive[EE] = np.maximum(state[EE], config.electron_energy_floor)
    return primitive.astype(np.float32)


def enforce_physical_floors(state: np.ndarray, config: SolverConfig = SolverConfig()) -> np.ndarray:
    """Clamp density/pressure/electron-energy to physically admissible values."""

    state = _ensure_state_shape(state).copy()
    rho = np.maximum(state[RHO], config.density_floor)
    prim = conserved_to_primitive(state, config=config)
    prim[RHO] = rho
    prim[ENERGY] = np.maximum(prim[ENERGY], config.pressure_floor)
    prim[EE] = np.maximum(prim[EE], config.electron_energy_floor)
    return primitive_to_conserved(
        prim,
        gamma=config.gamma,
        density_floor=config.density_floor,
        pressure_floor=config.pressure_floor,
    )


def estimate_timestep(state: np.ndarray, grid: CylindricalGrid, config: SolverConfig = SolverConfig()) -> float:
    """Estimate a stable explicit timestep from the fast magnetosonic speed."""

    prim = conserved_to_primitive(state, config=config).astype(np.float64)
    rho = np.maximum(prim[RHO], config.density_floor)
    pressure = np.maximum(prim[ENERGY], config.pressure_floor)
    b2 = prim[BR] ** 2 + prim[BZ] ** 2 + prim[BTH] ** 2
    bt_r = prim[BZ] ** 2 + prim[BTH] ** 2
    bt_z = prim[BR] ** 2 + prim[BTH] ** 2
    a2 = config.gamma * pressure / rho
    va2 = b2 / rho
    cf_r = np.sqrt(np.maximum(0.0, 0.5 * (a2 + va2 + np.sqrt(np.maximum((a2 - va2) ** 2 + 4.0 * a2 * bt_r / rho, 0.0)))))
    cf_z = np.sqrt(np.maximum(0.0, 0.5 * (a2 + va2 + np.sqrt(np.maximum((a2 - va2) ** 2 + 4.0 * a2 * bt_z / rho, 0.0)))))
    speed_r = np.max(np.abs(prim[MR]) + cf_r)
    speed_z = np.max(np.abs(prim[MZ]) + cf_z)
    return config.cfl / max(speed_r / grid.dr + speed_z / grid.dz, 1e-12)


def _apply_radial_bc(padded: np.ndarray, state: np.ndarray, current_I: float, grid: CylindricalGrid) -> None:
    nr, nz, ng = grid.nr, grid.nz, grid.ng
    padded[:, ng : ng + nr, ng : ng + nz] = state

    for g in range(ng):
        dst = ng - 1 - g
        src = ng + g
        padded[:, dst, ng : ng + nz] = padded[:, src, ng : ng + nz]
        padded[MR, dst, ng : ng + nz] = -padded[MR, src, ng : ng + nz]
        padded[MT, dst, ng : ng + nz] = -padded[MT, src, ng : ng + nz]
        padded[BR, dst, ng : ng + nz] = -padded[BR, src, ng : ng + nz]
        padded[BTH, dst, ng : ng + nz] = -padded[BTH, src, ng : ng + nz]

    for g in range(ng):
        dst = ng + nr + g
        src = ng + nr - 1
        padded[:, dst, ng : ng + nz] = padded[:, src, ng : ng + nz]
        padded[MR, dst, ng : ng + nz] = 0.0
        padded[BR, dst, ng : ng + nz] = 0.0
        r = grid.inner_radius + (nr + g) * grid.dr
        padded[BTH, dst, ng : ng + nz] = MU0 * current_I / (2.0 * math.pi * max(r, 1e-12))


def _apply_axial_bc(padded: np.ndarray, grid: CylindricalGrid) -> None:
    nr, nz, ng = grid.nr, grid.nz, grid.ng
    for g in range(ng):
        dst = ng - 1 - g
        src = ng + g
        padded[:, :, dst] = padded[:, :, src]
        padded[MZ, :, dst] = -padded[MZ, :, src]
        padded[BZ, :, dst] = -padded[BZ, :, src]

    for g in range(ng):
        dst = ng + nz + g
        src = ng + nz - 1
        padded[:, :, dst] = padded[:, :, src]


def pad_state(state: np.ndarray, grid: CylindricalGrid, current_I: float = 0.0) -> np.ndarray:
    """Apply radial electrode and axial reflecting/outflow boundary conditions."""

    state = _ensure_state_shape(state)
    padded = np.zeros((10, grid.nr + 2 * grid.ng, grid.nz + 2 * grid.ng), dtype=np.float32)
    _apply_radial_bc(padded, state, current_I, grid)
    _apply_axial_bc(padded, grid)
    return padded


def weno5z_left(stencil: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return left-biased WENO5-Z interface values from five-point stencils."""

    stencil = np.asarray(stencil, dtype=np.float64)
    s0, s1, s2, s3, s4 = [stencil[:, i, :] for i in range(5)]
    beta0 = (13.0 / 12.0) * (s0 - 2.0 * s1 + s2) ** 2 + 0.25 * (s0 - 4.0 * s1 + 3.0 * s2) ** 2
    beta1 = (13.0 / 12.0) * (s1 - 2.0 * s2 + s3) ** 2 + 0.25 * (s1 - s3) ** 2
    beta2 = (13.0 / 12.0) * (s2 - 2.0 * s3 + s4) ** 2 + 0.25 * (3.0 * s2 - 4.0 * s3 + s4) ** 2
    tau5 = np.abs(beta0 - beta2)
    d0, d1, d2 = 0.1, 0.6, 0.3
    r0 = np.minimum(tau5 / (beta0 + eps), 1e6)
    r1 = np.minimum(tau5 / (beta1 + eps), 1e6)
    r2 = np.minimum(tau5 / (beta2 + eps), 1e6)
    a0 = d0 * (1.0 + r0**2)
    a1 = d1 * (1.0 + r1**2)
    a2 = d2 * (1.0 + r2**2)
    asum = a0 + a1 + a2
    w0 = np.divide(a0, asum, out=np.full_like(a0, d0), where=asum > 0.0)
    w1 = np.divide(a1, asum, out=np.full_like(a1, d1), where=asum > 0.0)
    w2 = np.divide(a2, asum, out=np.full_like(a2, d2), where=asum > 0.0)
    p0 = (2.0 * s0 - 7.0 * s1 + 11.0 * s2) / 6.0
    p1 = (-s1 + 5.0 * s2 + 2.0 * s3) / 6.0
    p2 = (2.0 * s2 + 5.0 * s3 - s4) / 6.0
    return (w0 * p0 + w1 * p1 + w2 * p2).astype(np.float32)


def reconstruct_interfaces(padded: np.ndarray, axis: int, grid: CylindricalGrid) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct left/right states at cell interfaces using WENO5-Z."""

    if axis not in (1, 2):
        raise ValueError("axis must be 1 (r) or 2 (z)")
    ng = grid.ng
    nr = grid.nr
    nz = grid.nz
    if axis == 1:
        left = np.empty((10, nr + 1, nz), dtype=np.float32)
        right = np.empty_like(left)
        zs = slice(ng, ng + nz)
        for i in range(nr + 1):
            k = ng + i - 1
            st_left = padded[:, k - 2 : k + 3, zs]
            st_right = padded[:, k - 1 : k + 4, zs][:, ::-1, :]
            left[:, i, :] = weno5z_left(st_left)
            right[:, i, :] = weno5z_left(st_right)
        return left, right

    left = np.empty((10, nr, nz + 1), dtype=np.float32)
    right = np.empty_like(left)
    rs = slice(ng, ng + nr)
    for j in range(nz + 1):
        k = ng + j - 1
        st_left = np.transpose(padded[:, rs, k - 2 : k + 3], (0, 2, 1))
        st_right = np.transpose(padded[:, rs, k - 1 : k + 4], (0, 2, 1))[:, ::-1, :]
        left[:, :, j] = weno5z_left(st_left)
        right[:, :, j] = weno5z_left(st_right)
    return left, right


def _flux_divergence_radial(flux_r: np.ndarray, grid: CylindricalGrid) -> np.ndarray:
    ar = grid.radial_face_areas
    vol = grid.cell_volumes
    return ((flux_r[:, 1:, :] * ar[1:, None]) - (flux_r[:, :-1, :] * ar[:-1, None])) / vol[None, :, :]


def _flux_divergence_axial(flux_z: np.ndarray, grid: CylindricalGrid) -> np.ndarray:
    az = grid.axial_face_areas
    vol = grid.cell_volumes
    return ((flux_z[:, :, 1:] - flux_z[:, :, :-1]) * az[None, :, None]) / vol[None, :, :]


def compute_flux_divergence(
    state: np.ndarray,
    grid: CylindricalGrid,
    *,
    current_I: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite-volume radial/axial fluxes and total divergence."""

    padded = pad_state(state, grid, current_I=current_I)
    left_r, right_r = reconstruct_interfaces(padded, axis=1, grid=grid)
    left_z, right_z = reconstruct_interfaces(padded, axis=2, grid=grid)
    left_r = enforce_physical_floors(left_r)
    right_r = enforce_physical_floors(right_r)
    left_z = enforce_physical_floors(left_z)
    right_z = enforce_physical_floors(right_z)
    flux_r = hlld_flux_numpy(left_r, right_r, direction=0)
    flux_z = hlld_flux_numpy(left_z, right_z, direction=1)
    div = _flux_divergence_radial(flux_r, grid) + _flux_divergence_axial(flux_z, grid)
    return flux_r, flux_z, div.astype(np.float32)


def divergence_velocity(primitive: np.ndarray, grid: CylindricalGrid) -> np.ndarray:
    """Approximate cylindrical divergence of velocity."""

    r = grid.radii[:, None]
    vr = primitive[MR].astype(np.float64)
    vz = primitive[MZ].astype(np.float64)
    rvr = r * vr
    div_r = np.gradient(rvr, grid.dr, axis=0, edge_order=2) / np.maximum(r, 1e-12)
    div_z = np.gradient(vz, grid.dz, axis=1, edge_order=2)
    return (div_r + div_z).astype(np.float32)


def compute_current_density(state: np.ndarray, grid: CylindricalGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate current density components via axisymmetric curl(B)/mu0."""

    state = _ensure_state_shape(state).astype(np.float64)
    r = grid.radii[:, None]
    br = state[BR]
    bz = state[BZ]
    bth = state[BTH]
    jr = -np.gradient(bth, grid.dz, axis=1, edge_order=2) / MU0
    jz = np.gradient(r * bth, grid.dr, axis=0, edge_order=2) / np.maximum(r, 1e-12) / MU0
    jth = (np.gradient(br, grid.dz, axis=1, edge_order=2) - np.gradient(bz, grid.dr, axis=0, edge_order=2)) / MU0
    return jr.astype(np.float32), jz.astype(np.float32), jth.astype(np.float32)


def compute_source_terms(
    state: np.ndarray,
    grid: CylindricalGrid,
    *,
    config: SolverConfig = SolverConfig(),
) -> np.ndarray:
    """Return geometric, Ohmic, and bremsstrahlung source terms."""

    primitive = conserved_to_primitive(state, config=config)
    src = cylindrical_sources_numpy(primitive, grid.radii.astype(np.float32), dr=grid.dr).astype(np.float64)

    use_resistive = config.resistivity > 0.0 or config.use_spitzer_resistivity
    if use_resistive:
        jr, jz, jth = compute_current_density(state, grid)
        j2 = jr.astype(np.float64) ** 2 + jz.astype(np.float64) ** 2 + jth.astype(np.float64) ** 2
        if config.use_spitzer_resistivity:
            rho_sp = np.maximum(primitive[RHO].astype(np.float64), config.density_floor)
            ne_sp = rho_sp / M_DEUTERIUM
            Te_eV = _electron_temperature_eV(state, ne_sp, config)
            eta_local = spitzer_resistivity(
                Te_eV, Z=config.spitzer_Z, lnA=config.spitzer_lnA,
                eta_floor=config.spitzer_eta_floor, eta_cap=config.spitzer_eta_cap,
            )
            q_heat = eta_local * j2
        else:
            q_heat = config.resistivity * j2
        pressure = np.maximum(primitive[ENERGY].astype(np.float64), config.pressure_floor)
        rho = np.maximum(primitive[RHO].astype(np.float64), config.density_floor)
        entropy = state[SRHO].astype(np.float64) / rho
        src[ENERGY] += q_heat
        src[SRHO] += (config.gamma - 1.0) * entropy * q_heat / pressure
        src[EE] += q_heat

    if config.bremsstrahlung:
        rho = np.maximum(primitive[RHO].astype(np.float64), config.density_floor)
        ne = rho / M_DEUTERIUM
        te = (2.0 / 3.0) * np.maximum(state[EE].astype(np.float64), config.electron_energy_floor) / np.maximum(ne * K_B, 1e-30)
        q_brem = 1.42e-40 * ne**2 * np.sqrt(np.maximum(te, 0.0))
        src[ENERGY] -= q_brem
        src[EE] -= q_brem

    return src.astype(np.float32)


def rhs(state: np.ndarray, grid: CylindricalGrid, *, current_I: float = 0.0, config: SolverConfig = SolverConfig()) -> np.ndarray:
    """Semi-discrete MHD right-hand side."""

    _, _, div = compute_flux_divergence(state, grid, current_I=current_I)
    src = compute_source_terms(state, grid, config=config)
    return (-div + src).astype(np.float32)


def sound_speed(state: np.ndarray, config: SolverConfig = SolverConfig()) -> np.ndarray:
    """Adiabatic sound speed from the dual-energy primitive state."""

    primitive = conserved_to_primitive(state, config=config)
    return np.sqrt(config.gamma * np.maximum(primitive[ENERGY], config.pressure_floor) / np.maximum(primitive[RHO], config.density_floor))


def entropy_resynchronize(
    previous_state: np.ndarray,
    new_state: np.ndarray,
    grid: CylindricalGrid,
    *,
    config: SolverConfig = SolverConfig(),
) -> np.ndarray:
    """Reset entropy in shock cells where the total-energy pressure is reliable."""

    prev = _ensure_state_shape(previous_state).astype(np.float64)
    state = _ensure_state_shape(new_state).copy().astype(np.float64)
    primitive = conserved_to_primitive(state, config=config).astype(np.float64)
    div_v = divergence_velocity(primitive.astype(np.float32), grid).astype(np.float64)
    cs = sound_speed(state.astype(np.float32), config=config).astype(np.float64)
    p_s = np.maximum(state[SRHO] * np.maximum(state[RHO], config.density_floor) ** (config.gamma - 1.0), config.pressure_floor)
    rho = np.maximum(state[RHO], config.density_floor)
    ke = 0.5 * (state[MR] ** 2 + state[MZ] ** 2 + state[MT] ** 2) / rho
    me = 0.5 * (state[BR] ** 2 + state[BZ] ** 2 + state[BTH] ** 2)
    p_e = np.maximum((config.gamma - 1.0) * (state[ENERGY] - ke - me), config.pressure_floor)
    p_prev = recover_pressure(prev.astype(np.float32), gamma=config.gamma, eta1=config.eta_switch_low, eta2=config.eta_switch_high, density_floor=config.density_floor, pressure_floor=config.pressure_floor).astype(np.float64)
    pressure_jump = np.abs(p_e - p_prev) / np.maximum(p_s, config.pressure_floor)
    compression_limit = -config.entropy_sync_compression * cs / min(grid.dr, grid.dz)
    eta = p_s / np.maximum(np.abs(state[ENERGY]), 1e-30)
    energetic = p_e > 2.0 * p_s
    eta_gate = (eta > config.entropy_sync_eta) | energetic
    mask = ((div_v < compression_limit) | energetic) & (pressure_jump > config.entropy_sync_pressure_jump) & eta_gate
    state[SRHO][mask] = p_e[mask] / np.maximum(rho[mask] ** (config.gamma - 1.0), config.pressure_floor)
    return state.astype(np.float32)


def divergence_b(state: np.ndarray, grid: CylindricalGrid) -> np.ndarray:
    """Compute cell-centred divergence of the poloidal magnetic field."""

    state = _ensure_state_shape(state).astype(np.float64)
    r = grid.radii[:, None]
    rbr = r * state[BR]
    div_r = np.gradient(rbr, grid.dr, axis=0, edge_order=2) / np.maximum(r, 1e-12)
    div_z = np.gradient(state[BZ], grid.dz, axis=1, edge_order=2)
    return (div_r + div_z).astype(np.float32)


def constrained_transport_update(state: np.ndarray, grid: CylindricalGrid, dt: float, config: SolverConfig = SolverConfig()) -> np.ndarray:
    """Apply a lightweight CT-style correction using edge EMFs."""

    state = _ensure_state_shape(state).copy().astype(np.float64)
    primitive = conserved_to_primitive(state.astype(np.float32), config=config).astype(np.float64)
    r = grid.radii[:, None]
    e_theta = -(primitive[MR] * primitive[BZ] - primitive[MZ] * primitive[BR])
    d_e_dz = np.gradient(e_theta, grid.dz, axis=1, edge_order=2)
    d_re_dr = np.gradient(r * e_theta, grid.dr, axis=0, edge_order=2) / np.maximum(r, 1e-12)
    state[BR] -= dt * d_e_dz
    state[BZ] += dt * d_re_dr
    return state.astype(np.float32)


def thomas_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs_vec: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal linear system via the Thomas algorithm."""

    n = rhs_vec.size
    c = np.zeros(n - 1, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    c[0] = upper[0] / diag[0]
    d[0] = rhs_vec[0] / diag[0]
    for i in range(1, n - 1):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        c[i] = upper[i] / denom
        d[i] = (rhs_vec[i] - lower[i - 1] * d[i - 1]) / denom
    d[-1] = (rhs_vec[-1] - lower[-1] * d[-2]) / (diag[-1] - lower[-1] * c[-1])
    sol = np.empty(n, dtype=np.float64)
    sol[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        sol[i] = d[i] - c[i] * sol[i + 1]
    return sol


def _implicit_diffuse_axis(field: np.ndarray, alpha: float, axis: int) -> np.ndarray:
    """Implicitly diffuse a scalar field along one axis with zero-gradient ends."""

    if alpha <= 0.0:
        return field

    field = np.asarray(field, dtype=np.float64)
    out = np.empty_like(field)
    n = field.shape[axis]
    lower = np.full(n - 1, -alpha, dtype=np.float64)
    upper = np.full(n - 1, -alpha, dtype=np.float64)
    diag = np.full(n, 1.0 + 2.0 * alpha, dtype=np.float64)
    diag[0] = diag[-1] = 1.0 + alpha

    if axis == 0:
        for j in range(field.shape[1]):
            out[:, j] = thomas_solve(lower, diag, upper, field[:, j])
        return out

    for i in range(field.shape[0]):
        out[i, :] = thomas_solve(lower, diag, upper, field[i, :])
    return out


def implicit_resistive_diffusion(
    state: np.ndarray,
    grid: CylindricalGrid,
    dt: float,
    *,
    config: SolverConfig = SolverConfig(),
) -> np.ndarray:
    """Operator-split implicit diffusion for magnetic components."""

    if not config.enable_resistive_diffusion:
        return _ensure_state_shape(state)

    if config.use_spitzer_resistivity:
        s = _ensure_state_shape(state).astype(np.float64)
        rho = np.maximum(s[RHO], config.density_floor)
        ne = rho / M_DEUTERIUM
        Te_eV = _electron_temperature_eV(s, ne, config)
        eta_mean = float(np.mean(spitzer_resistivity(
            Te_eV, Z=config.spitzer_Z, lnA=config.spitzer_lnA,
            eta_floor=config.spitzer_eta_floor, eta_cap=config.spitzer_eta_cap,
        )))
    else:
        eta_mean = config.resistivity

    if eta_mean <= 0.0:
        return _ensure_state_shape(state)

    alpha_r = eta_mean * dt / (grid.dr * grid.dr)
    alpha_z = eta_mean * dt / (grid.dz * grid.dz)
    out = _ensure_state_shape(state).copy().astype(np.float64)
    for comp in (BR, BZ, BTH):
        tmp = _implicit_diffuse_axis(out[comp], alpha_z, axis=1)
        out[comp] = _implicit_diffuse_axis(tmp, alpha_r, axis=0)
    return out.astype(np.float32)


def extract_plasma_inductance(
    state: np.ndarray,
    grid: CylindricalGrid,
    circuit: CircuitParameters = PF1000_CIRCUIT,
    previous_inductance: Optional[float] = None,
) -> float:
    """Extract plasma inductance from the density distribution."""

    state = _ensure_state_shape(state).astype(np.float64)
    rho = np.maximum(state[RHO], 0.0)
    r = grid.radii[:, None]
    z = grid.axial
    sheath_profile = np.sum(rho * r * grid.dr, axis=0)
    z_sheath = float(z[int(np.argmax(sheath_profile))])
    weights = rho * grid.cell_volumes
    total_mass = max(float(np.sum(weights)), 1e-30)
    r_eff = float(np.sum(r * weights) / total_mass)
    r_eff = min(max(r_eff, circuit.anode_radius * 0.25), circuit.cathode_radius * 0.999)
    inductance = (MU0 / (2.0 * math.pi)) * z_sheath * math.log(circuit.cathode_radius / r_eff)
    if previous_inductance is not None:
        inductance = max(inductance, previous_inductance)
    return float(inductance)


def compute_back_emf(current: float, plasma_inductance: float, previous_inductance: float, dt: float) -> float:
    """Return the back-EMF ``-I * dLp/dt``."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return -float(current) * (float(plasma_inductance) - float(previous_inductance)) / dt


def step_circuit(
    circuit_state: CircuitState,
    params: CircuitParameters,
    dt: float,
    plasma_inductance: float,
) -> CircuitState:
    """Advance the external RLC circuit one step in float64."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    lp_prev = float(circuit_state.plasma_inductance)
    lp = max(float(plasma_inductance), lp_prev)
    dlp_dt = (lp - lp_prev) / dt
    l_total = params.L0 + lp
    di_dt = (circuit_state.capacitor_voltage - (params.R0 + dlp_dt) * circuit_state.current) / max(l_total, 1e-30)
    dv_dt = -circuit_state.current / max(params.C, 1e-30)
    current_new = float(circuit_state.current + dt * di_dt)
    voltage_new = float(circuit_state.capacitor_voltage + dt * dv_dt)
    return CircuitState(
        current=current_new,
        capacitor_voltage=voltage_new,
        plasma_inductance=lp,
        time=float(circuit_state.time + dt),
        back_emf=compute_back_emf(circuit_state.current, lp, lp_prev, dt),
    )


class MHDSolver:
    """High-level dual-energy cylindrical MHD solver."""

    def __init__(
        self,
        grid: CylindricalGrid,
        *,
        config: SolverConfig = SolverConfig(),
        circuit: Optional[CircuitParameters] = None,
    ) -> None:
        self.grid = grid
        self.config = config
        self.circuit = circuit

    def courant_timestep(self, state: np.ndarray) -> float:
        return estimate_timestep(state, self.grid, self.config)

    def stage_rhs(self, state: np.ndarray, current_I: float = 0.0) -> np.ndarray:
        return rhs(state, self.grid, current_I=current_I, config=self.config)

    def _post_process(self, state: np.ndarray, previous_state: np.ndarray, dt: float) -> np.ndarray:
        out = entropy_resynchronize(previous_state, state, self.grid, config=self.config)
        if self.config.enable_ct:
            out = constrained_transport_update(out, self.grid, dt, config=self.config)
        out = implicit_resistive_diffusion(out, self.grid, dt, config=self.config)
        return enforce_physical_floors(out, self.config)

    def _step_ssp_rk3(
        self,
        state: np.ndarray,
        dt: float,
        circuit_state: Optional[CircuitState] = None,
    ) -> tuple[np.ndarray, Optional[CircuitState]]:
        """Advance one stable SSP-RK3 substep."""

        state0 = enforce_physical_floors(state, self.config)
        current_I = 0.0 if circuit_state is None else circuit_state.current

        if self.circuit is not None and circuit_state is not None:
            lp = extract_plasma_inductance(state0, self.grid, self.circuit, circuit_state.plasma_inductance)
            circuit_state = step_circuit(circuit_state, self.circuit, dt, lp)
            current_I = circuit_state.current

        k1 = self.stage_rhs(state0, current_I=current_I)
        u1 = self._post_process(state0 + dt * k1, state0, dt)

        k2 = self.stage_rhs(u1, current_I=current_I)
        u2 = self._post_process(0.75 * state0 + 0.25 * (u1 + dt * k2), u1, dt)

        k3 = self.stage_rhs(u2, current_I=current_I)
        un = self._post_process((state0 + 2.0 * (u2 + dt * k3)) / 3.0, u2, dt)
        return un.astype(np.float32), circuit_state

    def step(
        self,
        state: np.ndarray,
        dt: float,
        circuit_state: Optional[CircuitState] = None,
    ) -> tuple[np.ndarray, Optional[CircuitState]]:
        """Advance the plasma and optional circuit, subcycling as needed."""

        state = _ensure_state_shape(state)
        remaining = float(dt)
        while remaining > 0.0:
            local_dt = min(remaining, max(1e-12, self.courant_timestep(state)))
            state, circuit_state = self._step_ssp_rk3(state, local_dt, circuit_state)
            remaining -= local_dt
        return state, circuit_state


def make_uniform_primitive(
    grid: CylindricalGrid,
    *,
    rho: float,
    pressure: float,
    vr: float = 0.0,
    vz: float = 0.0,
    vtheta: float = 0.0,
    br: float = 0.0,
    bz: float = 0.0,
    btheta: float | np.ndarray = 0.0,
    electron_fraction: float = 0.5,
) -> np.ndarray:
    """Create a uniform primitive state on the provided grid."""

    prim = np.zeros((10, grid.nr, grid.nz), dtype=np.float32)
    prim[RHO] = rho
    prim[MR] = vr
    prim[MZ] = vz
    prim[MT] = vtheta
    prim[ENERGY] = pressure
    prim[SRHO] = pressure / max(rho**GAMMA, 1e-30)
    prim[BR] = br
    prim[BZ] = bz
    prim[BTH] = np.asarray(btheta, dtype=np.float32)
    prim[EE] = electron_fraction * pressure / (GAMMA - 1.0)
    return prim


def initialize_pf1000_state(
    grid: CylindricalGrid,
    *,
    circuit: CircuitParameters = PF1000_CIRCUIT,
    fill_pressure: float = 160.0,
    temperature: float = 300.0,
    sheath_density_ratio: float = 100.0,
) -> np.ndarray:
    """Construct a simple PF-1000 initial state with a dense current sheath."""

    rho0 = fill_pressure * M_DEUTERIUM / (K_B * temperature)
    prim = make_uniform_primitive(grid, rho=rho0, pressure=fill_pressure)
    r = grid.radii[:, None]
    z = grid.axial[None, :]
    z0 = grid.axial[int(0.75 * (grid.nz - 1))]
    sigma_z = max(4.0 * grid.dz, 0.05 * grid.nz * grid.dz)
    sigma_r = max(2.0 * grid.dr, 0.1 * circuit.anode_radius)
    sheath = np.exp(-((z - z0) ** 2) / (2.0 * sigma_z**2)) * np.exp(-((r - circuit.anode_radius) ** 2) / (2.0 * sigma_r**2))
    prim[RHO] *= 1.0 + sheath_density_ratio * sheath
    prim[ENERGY] *= 1.0 + 0.1 * sheath
    return primitive_to_conserved(prim)


def run_pf1000_simulation(
    *,
    total_time: float = 12e-6,
    dt: float = 1e-7,
    nr: int = 64,
    nz: int = 128,
    length: float = 0.45,
    circuit: CircuitParameters = PF1000_CIRCUIT,
    config: SolverConfig = SolverConfig(),
    initial_state: Optional[np.ndarray] = None,
) -> PF1000RunResult:
    """Run a PF-1000 discharge using the current solver scaffold."""

    grid = CylindricalGrid(nr=nr, nz=nz, dr=circuit.cathode_radius / nr, dz=length / nz)
    solver = MHDSolver(grid, config=config, circuit=circuit)
    state = initialize_pf1000_state(grid, circuit=circuit) if initial_state is None else _ensure_state_shape(initial_state)
    lp0 = extract_plasma_inductance(state, grid, circuit)
    circuit_state = CircuitState(current=0.0, capacitor_voltage=circuit.V0, plasma_inductance=lp0, time=0.0, back_emf=0.0)
    nsteps = max(1, int(math.ceil(total_time / dt)))
    times = np.empty(nsteps + 1, dtype=np.float64)
    currents = np.empty(nsteps + 1, dtype=np.float64)
    voltages = np.empty(nsteps + 1, dtype=np.float64)
    inductances = np.empty(nsteps + 1, dtype=np.float64)
    sheath_positions = np.empty(nsteps + 1, dtype=np.float64)
    radiated_energy = np.zeros(nsteps + 1, dtype=np.float64)
    times[0] = circuit_state.time
    currents[0] = circuit_state.current
    voltages[0] = circuit_state.capacitor_voltage
    inductances[0] = circuit_state.plasma_inductance
    sheath_positions[0] = extract_sheath_position(state, grid)

    for n in range(1, nsteps + 1):
        state, circuit_state = solver.step(state, dt, circuit_state)
        assert circuit_state is not None
        times[n] = circuit_state.time
        currents[n] = circuit_state.current
        voltages[n] = circuit_state.capacitor_voltage
        inductances[n] = circuit_state.plasma_inductance
        sheath_positions[n] = extract_sheath_position(state, grid)
        radiated_energy[n] = radiated_energy[n - 1] + _estimate_radiated_power(state, grid, config) * dt

    dI_dt = np.gradient(currents, times)
    stored_energy = 0.5 * circuit.C * circuit.V0**2

    return PF1000RunResult(
        times=times,
        currents=currents,
        voltages=voltages,
        inductances=inductances,
        state=state,
        dI_dt=dI_dt,
        sheath_positions=sheath_positions,
        radiated_energy=radiated_energy,
        stored_energy=stored_energy,
    )


def validate_pf1000(
    result: PF1000RunResult,
    targets: Optional[dict[str, "ValidationTarget"]] = None,
) -> dict[str, tuple[float, bool]]:
    """Compare simulation results against all 8 PF-1000 validation targets.

    Returns a mapping from target name to ``(measured_value, passes)`` where
    *passes* is ``True`` when the measured value falls within the reference
    range.
    """
    if targets is None:
        targets = PF1000_VALIDATION_TARGETS

    metrics: dict[str, float] = {
        "peak_current": result.peak_current,
        "time_of_peak_current": result.time_of_peak_current,
        "current_dip_fraction": result.current_dip_fraction,
        "pinch_time": result.pinch_time,
        "peak_dI_dt": result.peak_dI_dt,
        "inductance_at_pinch": result.inductance_at_pinch,
        "mean_sheath_speed": result.mean_sheath_speed,
        "radiated_energy_fraction": result.radiated_energy_fraction,
    }

    report: dict[str, tuple[float, bool]] = {}
    for name, target in targets.items():
        value = metrics.get(name, float("nan"))
        passes = target.reference_low <= value <= target.reference_high
        report[name] = (value, passes)
    return report
