import numpy as np
import os
import sys
import heapq
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class Atom:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def __eq__(self, other):
        if isinstance(other, Atom):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False
    def add(self, delta):
        return Atom(self.x+delta[0],self.y+delta[1],self.z+delta[2] )
    def __lt__(self, other):
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)
    def getPos(self):
        return (self.x, self.y, self.z)
    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __repr__(self):
        return f"Atom({self.x}, {self.y}, {self.z})"
    
class UnitCell:
    @staticmethod
    def normalize(a):
        a = np.asarray(a)
        return a / np.linalg.norm(a)
    @staticmethod
    def _sublattice(atom):    
        return (round(atom.x*4) + round(atom.y*4) + round(atom.z*4)) & 1
    @staticmethod
    def directionalCosine(delta):
        dx,dy,dz = delta
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if norm != 0:
            l = dx / norm
            m = dy / norm
            n = dz / norm
        return l, m, n
    @staticmethod
    def createRotationMatrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    
    raw_deltas = {
            0: [(+0.25, +0.25, +0.25), (+0.25, -0.25, -0.25),
                (-0.25, +0.25, -0.25), (-0.25, -0.25, +0.25)],
            1: [(-0.25, -0.25, -0.25), (-0.25, +0.25, +0.25),
                (+0.25, -0.25, +0.25), (+0.25, +0.25, -0.25)]
        }
    
    
    orientation_to_start ={
        (0,1,2,3) : Atom(0,0,0),
        (1,2,3,0) : Atom(0,0.25,0.25),
        (2,3,0,1) : Atom(0,0.5,0),
        (3,0,1,2) : Atom(0,0.75, 0.25)
    } 


    @staticmethod
    def determine_hybridization(delta):
        sign_pattern = np.sign(delta)
        if np.array_equal(sign_pattern, [1, 1, 1]):       # Type a
            return 0
        elif np.array_equal(sign_pattern, [1, -1, -1]):   # Type b
            return 1
        elif np.array_equal(sign_pattern, [-1, 1, -1]):   # Type c
            return 2
        elif np.array_equal(sign_pattern, [-1, -1, 1]):   # Type d
            return 3
    
    a = 0.5431e-9
    def __init__(self, channel_length : float, channel_width : float, channel_thickness : float, orientation = (0,1,2,3), not_NEGF = False):
        """
        Builds the nanowire unit cell. 
        
        """
       
        self.channel_length = channel_length
        self.channel_width = channel_width
        self.channel_thickness = channel_thickness 
        
        self.Nz = round(channel_thickness / self.a * 100) /100
        self.Nx = round(self.channel_length / self.a * 100) / 100
        self.Ny = round(self.channel_width / self.a * 100) / 100
   
        self.ATOM_POSITIONS = [] # the order of atoms in this cell is very important as this governs the layer structure 
        self.neighbors = {}
        self.danglingBonds = {}
        self.addAtoms(UnitCell.orientation_to_start[orientation])
        self.map_neighbors_and_dangling_bonds()
    def check_in_z_direction(self, atom : Atom):
        return not (atom.z >= self.Nz or atom.z < 0)
    def check_in_x_direction(self, atom : Atom):
        return not (atom.x >= self.Nx or atom.x < 0) 
    def check_in_y_direction(self, atom : Atom):
        return not (atom.y >= self.Ny or atom.y < 0) 
    def addAtoms(self, start_atom: Atom):
        """Priority queue traversal to fill ATOM_POSITIONS in (x, y, z) order."""
        import heapq
        visited = set()
        heap = []
        in_heap = set()
        heapq.heappush(heap, (start_atom.x, start_atom.y, start_atom.z, start_atom))
        in_heap.add(start_atom)

        while heap:
            x, y, z, atom = heapq.heappop(heap)
            in_heap.remove(atom)
            if atom in visited:
                continue
            visited.add(atom)
            self.ATOM_POSITIONS.append(atom)
            for neighbor in self.get_neighbors(atom):
                if neighbor not in visited and neighbor not in in_heap and self.check_in_x_direction(neighbor) and self.check_in_y_direction(neighbor) and self.check_in_z_direction(neighbor):
                    heapq.heappush(heap, (neighbor.x, neighbor.y, neighbor.z, neighbor))
                    in_heap.add(neighbor)
        # Ensure strict (x, y, z) order
        self.ATOM_POSITIONS.sort(key=lambda atom: (atom.x, atom.y, atom.z))
    def map_neighbors_and_dangling_bonds(self):
        """
        Populates self.neighbors and self.danglingBonds for all atoms in ATOM_POSITIONS.
        - self.neighbors[atom]: list of (neighbor, delta, l, m, n) for valid neighbors inside the cell
        - self.danglingBonds[atom]: list of (neighbor, hybridization) for neighbors outside in Y or Z
        """
        self.neighbors = {}
        self.danglingBonds = {}
        atom_set = set(self.ATOM_POSITIONS)
        for atom in self.ATOM_POSITIONS:
            sublattice = UnitCell._sublattice(atom)
            self.neighbors[atom] = []
            self.danglingBonds[atom] = []
            for delta in UnitCell.raw_deltas[sublattice]:
                neighbor = atom.add(delta)
                # Ignore dangling bonds in X direction
                if not self.check_in_y_direction(neighbor) or not self.check_in_z_direction(neighbor):
                    self.danglingBonds[atom].append((neighbor, UnitCell.determine_hybridization(delta)))
                elif self.check_in_x_direction(neighbor):
                    # Only add as neighbor if inside cell in all directions
                    if neighbor in atom_set:
                        l, m, n = UnitCell.directionalCosine(delta)
                        self.neighbors[atom].append((neighbor, delta, l, m, n))
                    
    def get_neighbors(self, atom: Atom):
        """Return a list of neighboring Atom objects using raw_deltas and sublattice."""
        sublattice = UnitCell._sublattice(atom)
        deltas = UnitCell.raw_deltas[sublattice]
        neighbors = [atom.add(delta) for delta in deltas]
        return neighbors
        
        