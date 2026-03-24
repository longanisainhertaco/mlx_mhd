# mlx_mhd

Custom Metal kernels for a cylindrical MHD solver using [MLX](https://github.com/ml-explore/mlx).

## Contents
- Ghost-cell padding kernel with electrode boundary conditions
- HLLD Riemann solver kernel (entropy tracer supported)
- Cylindrical geometric source-term kernel
- Python wrappers plus slow NumPy reference functions

All arrays are float32 in structure-of-arrays layout `(10, nr, nz)` with components:
`[rho, vr, vz, vtheta, p, Srho, Br, Bz, Btheta, Ee]`.

## Usage (Apple Silicon + MLX)
```python
import mlx_mhd as mhd
import mlx.core as mx

state = mx.zeros((10, 256, 512), dtype=mx.float32)
pad = mhd.ghost_pad(state, current_I=200.0, dr=1.0/256)

left = state[:, :-1]
right = state[:, 1:]
flux = mhd.hlld_flux(left, right)

radii = mx.linspace(0.5/256, 1.0 - 0.5/256, 256)
sources = mhd.cylindrical_sources(state, radii)
```

Reference CPU versions are available as `ghost_pad_numpy`, `hlld_flux_numpy`, and
`cylindrical_sources_numpy` for testing or debugging.

## Threadgroup sizing (M3 Pro, 14 GPU cores)
For best occupancy, launch around 256 threads per group:
- Ghost padding: `(tg_r, tg_z, tg_comp) = (16, 8, 4)` with grid `(nr+2*ng, nz, 10)`
- HLLD flux and geometric sources: `(16, 8)` works well for `(nr, nz)` grids

You can retrieve the suggested padding size via `recommended_thread_group(nr, nz)`.
