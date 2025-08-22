"""Utility functions to generate standard meshes and boundary condition markers
for 1D / 2D diode and simple 2D transistor style geometries using dolfinx.

Geometries
----------
1. 1D Diode (interval) : length L, Nx elements
   Facet markers: 1 -> left (anode), 2 -> right (cathode)

2. 2D Diode (rectangle) : length L (x), width W (y), Nx * Ny mesh
   Facet markers: 1 -> left, 2 -> right, 3 -> bottom (e.g. substrate), 4 -> top

3. 2D Transistor (planar) simplified cross‑section:
   Regions along x: Source (Ls), Gate (Lg), Drain (Ld), total length L = Ls+Lg+Ld
   Vertical layering: Semiconductor of thickness Tch topped by Oxide tox (optional)
   Cell region markers (MeshTags on cells):
       1 -> semiconductor, 2 -> oxide (if tox>0)
   Facet markers: 1 left/source contact, 2 right/drain contact,
                  3 bottom substrate, 4 top gate oxide surface (oxide top if oxide present else semiconductor top)

Returned Objects
----------------
Each creator returns (mesh, cell_tags, facet_tags) where tags are dolfinx.mesh.meshtags
or None if not applicable.  Tag data can be used to build Dirichlet or Neumann BCs.

Helper
------
build_dirichlet_bcs(V, facet_tags, values) -> list of dolfinx.fem.dirichletbc
    values: dict mapping marker(int) -> float (potential in Volts) OR Constant
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

try:
    import dolfinx
    from mpi4py import MPI
    import ufl  # noqa: F401
    from petsc4py import PETSc
except ImportError as e:  # pragma: no cover
    raise RuntimeError("dolfinx related packages must be installed to use mesh_creator") from e


def _facet_tags_interval(mesh):
    """Mark end facets of 1D interval with ids 1 (x=min) and 2 (x=max)."""
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    fspace = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], mesh.geometry.x.min()) | np.isclose(x[0], mesh.geometry.x.max()))
    values = []
    for f in fspace:
        x = dolfinx.mesh.compute_midpoints(mesh, tdim - 1, np.array([f], dtype=np.int32))[0]
        if np.isclose(x[0], mesh.geometry.x.min()):
            values.append(1)
        else:
            values.append(2)
    ft = dolfinx.mesh.meshtags(mesh, tdim - 1, fspace, np.array(values, dtype=np.int32))
    return ft


def create_diode_1d(length: float, nx: int) -> Tuple[dolfinx.mesh.Mesh, None, dolfinx.mesh.MeshTags]:
    """Create 1D diode mesh."""
    mesh = dolfinx.mesh.create_interval(MPI.COMM_WORLD, nx, [0.0, length])
    facet_tags = _facet_tags_interval(mesh)
    return mesh, None, facet_tags


def _facet_tags_rectangle(mesh, Lx, Ly):
    tdim = mesh.topology.dim
    def marker_left(x):
        return np.isclose(x[0], 0.0)
    def marker_right(x):
        return np.isclose(x[0], Lx)
    def marker_bottom(x):
        return np.isclose(x[1], 0.0)
    def marker_top(x):
        return np.isclose(x[1], Ly)
    facets_left = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, marker_left)
    facets_right = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, marker_right)
    facets_bottom = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, marker_bottom)
    facets_top = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, marker_top)
    facets = np.concatenate([facets_left, facets_right, facets_bottom, facets_top])
    values = np.concatenate([
        np.full(facets_left.size, 1, dtype=np.int32),
        np.full(facets_right.size, 2, dtype=np.int32),
        np.full(facets_bottom.size, 3, dtype=np.int32),
        np.full(facets_top.size, 4, dtype=np.int32)
    ])
    order = np.argsort(facets)
    facets = facets[order]; values = values[order]
    facet_tags = dolfinx.mesh.meshtags(mesh, tdim - 1, facets, values)
    return facet_tags


def create_diode_2d(length: float, width: float, nx: int, ny: int) -> Tuple[dolfinx.mesh.Mesh, None, dolfinx.mesh.MeshTags]:
    """Create 2D rectangular diode mesh with facet markers."""
    p0 = [0.0, 0.0]; p1 = [length, width]
    mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [p0, p1], [nx, ny], cell_type=dolfinx.mesh.CellType.triangle)
    facet_tags = _facet_tags_rectangle(mesh, length, width)
    return mesh, None, facet_tags


def create_transistor_2d(Ls: float, Lg: float, Ld: float, W: float, tox: float, nx: int, ny: int, ny_ox: Optional[int] = None):
    """Create a simplistic 2D transistor cross-section mesh with optional oxide layer.

    Geometry: Source|Gate|Drain along x, semiconductor thickness W in y, oxide layer of thickness tox on top.
    Cell markers: 1 semiconductor, 2 oxide (if tox>0)
    Facet markers: 1 left/source, 2 right/drain, 3 bottom substrate, 4 top gate surface (oxide top if oxide present else semiconductor top)
    """
    L = Ls + Lg + Ld
    if tox < 0:
        raise ValueError("tox must be >= 0")
    # Build a simple rectangular mesh covering full height (W + tox)
    total_height = W + (tox if tox > 0 else 0.0)
    p0 = [0.0, 0.0]; p1 = [L, total_height]
    ny_total = ny + (ny_ox if (tox > 0 and ny_ox) else (ny if tox > 0 else 0))
    mesh = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [p0, p1], [nx, ny_total], cell_type=dolfinx.mesh.CellType.triangle)
    tdim = mesh.topology.dim

    # Cell tagging: classify by y coordinate
    mesh.topology.create_connectivity(tdim - 1, tdim)
    num_cells = mesh.topology.index_map(tdim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, cell_indices)
    cell_values = np.ones(num_cells, dtype=np.int32)  # default semiconductor
    if tox > 0:
        oxide_mask = midpoints[:, 1] >= W
        cell_values[oxide_mask] = 2
    cell_tags = dolfinx.mesh.meshtags(mesh, tdim, cell_indices, cell_values)

    # Facet tags reuse rectangle helper with total height
    facet_tags = _facet_tags_rectangle(mesh, L, total_height)
    return mesh, cell_tags, facet_tags


def build_dirichlet_bcs(V, facet_tags, values: Dict[int, float | PETSc.ScalarType]):
    """Construct Dirichlet BCs on a scalar function space V given facet marker -> potential mapping.
    values: mapping facet_id -> potential (float or PETSc scalar).
    Returns list of dolfinx.fem.DirichletBC objects.
    """
    bcs = []
    for marker, val in values.items():
        dofs = dolfinx.fem.locate_dofs_topological(V, V.mesh.topology.dim - 1, facet_tags.find(marker))
        if not isinstance(val, dolfinx.fem.Constant):
            val = dolfinx.fem.Constant(V.mesh, PETSc.ScalarType(val))
        bcs.append(dolfinx.fem.dirichletbc(val, dofs, V))
    return bcs


__all__ = [
    'create_diode_1d', 'create_diode_2d', 'create_transistor_2d',
    'build_dirichlet_bcs'
]
