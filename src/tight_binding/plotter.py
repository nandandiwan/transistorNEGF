import numpy as np
import matplotlib.pyplot as plt
from finite_tight_binding import TightBindingHamiltonian

class BandStructurePlotter():
    def __init__(self, TB_Object : TightBindingHamiltonian):
        self.N = TB_Object.N
        self.TB = TB_Object
        self.unitCell = self.TB.unitCell
        
        self.k_path = None
        self.k_ticks = None      
        self.energies = None    

    # k space
    @staticmethod
    def _cumulative_distance(k_path):
        d = np.zeros(len(k_path))
        for i in range(1, len(k_path)):
            d[i] = d[i-1] + np.linalg.norm(k_path[i] - k_path[i-1])
        return d

    @staticmethod
    def _segment(a, b, n):
        """n points from a→b (excluding b)."""
        return np.linspace(a, b, n, endpoint=False)

    def build_k_path(self, corner_points, points_per_segment=200):
        """
        corner_points : list of 3-vectors (in Cartesian reciprocal space)
            e.g. [W, Γ, K, …].
        """
        k = []
        for p, q in zip(corner_points[:-1], corner_points[1:]):
            k.extend(self._segment(p, q, points_per_segment))
        k.append(corner_points[-1])          # include the last corner exactly
        self.k_path = np.array(k)
        self.k_ticks = {lbl: i*points_per_segment
                        for i, lbl in enumerate("".join((" " * (len(corner_points)-1))).split())}

    #eigen states
    def _compute_band_structure(self):
        assert self.k_path is not None, "call build_k_path() first"
        Nk = len(self.k_path)
        # get one eigen-spectrum just to know how many bands
        test_E = self.TB.create_tight_binding(self.k_path[0], self.N)[0]
        Nb = test_E.size
        self.energies = np.empty((Nk, Nb))
        for i, kvec in enumerate(self.k_path):
            self.energies[i] = np.sort(self.TB.create_tight_binding(kvec, self.N)[0].real)

    # plot
    def plot(self, energy_window=None, colour_cycle=None,
             k_labels=None, linewidth=1.2, figsize=(8, 6)):
        """
        energy_window : (Emin, Emax) tuple or None
            Energies outside are masked out (useful to hide surface bands).
        k_labels : dict {label: index} or None
            Override tick labels.  If None, W-Γ-K… order is auto-generated.
        """
        if self.energies is None:
            self._compute_band_structure()

        # x-axis
        k_x = self._cumulative_distance(self.k_path)

        # optional masking
        E = self.energies.copy()
        if energy_window is not None:
            Emin, Emax = energy_window
            E[(E < Emin) | (E > Emax)] = np.nan

        # plotting
        plt.figure(figsize=figsize)
        if colour_cycle:
            plt.gca().set_prop_cycle(color=colour_cycle)

        for band in range(E.shape[1]):
            plt.plot(k_x, E[:, band], lw=linewidth)

        # decorations
        plt.ylabel("Energy (eV)")
        plt.xlabel("$|\\mathbf{k}|$")
        plt.title("Band structure")
        plt.grid(True, ls="--", lw=0.4, alpha=0.5)

        # ticks at high-symmetry points
        if k_labels is None and self.k_ticks:
            k_labels = self.k_ticks
        if k_labels:
            tick_pos = [k_x[idx] for idx in k_labels.values()]
            for x in tick_pos:
                plt.axvline(x, color='grey', ls='--', lw=0.6)
            plt.xticks(tick_pos, list(k_labels.keys()))

        plt.tight_layout()
        plt.show()
        
        self.k_path = None
        self.k_ticks = None      
        self.energies = None  
        
    
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

class PotentialPlotter:
    def __init__(self, TB_Object):
        self.unitCell = TB_Object.unitCell          # no other state needed

    def plot_V_vs_z(self):
  
        V   = self.unitCell.voltageProfile          # shape (Nx, Ny, Nz)
        print(V)
        mean_V = V.mean(axis=(0, 1))                # (Nz,)
        z_axis = np.arange(len(mean_V))             # 0 … Nz‑1

        plt.figure(figsize=(6, 4))
        plt.plot(z_axis, mean_V, marker='o', lw=1.5)
        plt.xlabel("z index")
        plt.ylabel(r"$\langle V\rangle_{xy}$  (V)")
        plt.grid(alpha=0.4)
        plt.title("Average potential vs z")
        plt.tight_layout()
        plt.show()
