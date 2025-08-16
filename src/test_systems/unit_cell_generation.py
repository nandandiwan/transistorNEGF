import numpy as np
import os
import sys
import heapq
import scipy.sparse as sp

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
import numpy as np
import os
import sys
import heapq
import scipy.sparse as sp

@staticmethod
def normalize(a):
    a = np.asarray(a)
    return a / np.linalg.norm(a)
@staticmethod
def directionalCosine(delta):
    dx,dy,dz = delta
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    
    if norm != 0:
        l = dx / norm
        m = dy / norm
        n = dz / norm
    return l, m, n
class Atom:
    def __init__(self, x, y, z, precision=10):
        self.x = round(x, precision)
        self.y = round(y, precision)
        self.z = round(z, precision)


    def add(self, delta):
        new_x = self.x + delta[0]
        new_y = self.y + delta[1]
        new_z = self.z + delta[2]

        return Atom(new_x, new_y, new_z)

    def __eq__(self, other):
        return isinstance(other, Atom) and self.pos() == other.pos()

    def __hash__(self):
        return hash(self.pos())

    def pos(self):
        """Return the position as a tuple for hashing and comparison."""
        return (self.x, self.y, self.z)

    def add(self, delta):
        return Atom(self.x + delta[0], self.y + delta[1], self.z + delta[2])

    def __lt__(self, other):
        return self.pos() < other.pos()
        
    def __repr__(self):
        return f"Atom({self.x}, {self.y}, {self.z})"
    
class GrapehenearmchairCell:
    a = 1
    sin60 = np.sqrt(3) / 2
    cos60 = 1/2 
    base = [Atom(sin60,0,0), Atom(0, cos60, 0), Atom(0,1.5,0), Atom(sin60, 2,0)]
    
    deltas = {
        0 : [(0,1,0), (-sin60, -cos60, 0),(sin60, -cos60, 0)],
        1 : [(0,-1,0), (-sin60, cos60, 0),(sin60, cos60, 0)]
    }
    def __init__(self, num_layers_x = 10, num_layers_y = 5, periodic = False):
        self.num_layers_y = num_layers_y
        self.num_layers_x = num_layers_x
        self.periodic = periodic
        if self.periodic:
            self.num_layers_y = 1
        
        self.max_X = self.sin60 + num_layers_x *2*self.sin60
        self.max_Y = 3 *num_layers_y - 1
         
        
        self.layer = self.create_first_layer()
        self.structure = self.create_full_structure()
        
        self.sublattice = {}
        
        for idx, atom in enumerate(self.structure):
            self.sublattice[atom] = (idx+1)%2
        
        self.neighbors, self.dangling = self.neighbors_and_dangling()
        self.atom_to_idx = {atom: i for i, atom in enumerate(self.structure)}
        self.idx_to_atom = {i: atom for i, atom in enumerate(self.structure)}
    def create_first_layer(self):
        layer = []
        for y in range(self.num_layers_y):
            delta = (0,3*y, 0)
            new_atoms = map(lambda x: x.add(delta), GrapehenearmchairCell.base)
            layer.extend(new_atoms)
        
        return layer
    
    def create_full_structure(self):
        structure = []
        for x in range(self.num_layers_x):
            delta = (self.sin60 * x*2, 0,0)
            new_atoms = map(lambda x: x.add(delta), self.layer)
            structure.extend(new_atoms)
        
        return structure
    
    def check_in_y_direction(self, atom):
        if (atom.y < 0 or atom.y > self.max_Y):
            return False
        return True
    def check_in_x_direction(self, atom):
        if (atom.x < 0 or atom.x > self.max_X):
            return False
        return True
    def directionalCosine(delta):
        dx,dy,dz = delta
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if norm != 0:
            l = dx / norm
            m = dy / norm
            n = dz / norm
        return l, m, n
    def get_sublattice(self, atom: Atom):
        """return 0 or 1"""
        return self.sublattice[atom]
    def neighbors_and_dangling(self):
        neighbors = {}
        dangling_bonds = {}
        atom_set = set(self.structure)
        
        for atom in self.structure:
            neighbors[atom] = []
            dangling_bonds[atom] = []
            deltas = GrapehenearmchairCell.deltas[self.get_sublattice(atom)]
            if (self.periodic):
                for delta in deltas:
                    neighbor = atom.add(delta)
                    # there are no dangling bonds i think
                    if (not self.check_in_y_direction(neighbor)):
                        l, m, n = GrapehenearmchairCell.directionalCosine(delta)
                        # periodic y value
                        neighbor.y = neighbor.y % 3
                        neighbors[atom].append((neighbor, delta, l, m, n))   
                    elif (self.check_in_y_direction(neighbor) and self.check_in_x_direction(neighbor)):
                        # in the structure 
                        l, m, n = GrapehenearmchairCell.directionalCosine(delta)
                        neighbors[atom].append((neighbor, delta, l, m, n))   
                        
            else:
                for delta in deltas:
                    neighbor = atom.add(delta)
                    
                    if (not self.check_in_y_direction(neighbor)):
                        l, m, n = GrapehenearmchairCell.directionalCosine(delta)
                        dangling_bonds[atom].append((neighbor, delta, l, m, n))  
                    elif (self.check_in_y_direction(neighbor) and self.check_in_x_direction(neighbor)):

                        # in the structure 
                        l, m, n = GrapehenearmchairCell.directionalCosine(delta)
                        neighbors[atom].append((neighbor, delta, l, m, n))   
                    # ignore x 
        
        return neighbors, dangling_bonds


class SiliconUnitCell:
    """unit cell of silicon, periodic in y or not"""
    orientation_to_start ={
        (0,1,2,3) : Atom(0,0,0),
        (1,2,3,0) : Atom(0,0.25,0.25),
        (2,3,0,1) : Atom(0,0.5,0),
        (3,0,1,2) : Atom(0,0.75, 0.25),
        (0,3,2,1) : Atom(0,0,0),
        (1,0,3,1) : Atom(0,0.25,0.25),
        (2,1,0,3) : Atom(0,0.5,0),
        (3,2,1,0) : Atom(0,0.75, 0.25)
    } 
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
    

    
    
    orientation_to_start ={
        (0,1,2,3) : Atom(0,0,0),
        (1,2,3,0) : Atom(0,0.25,0.25),
        (2,3,0,1) : Atom(0,0.5,0),
        (3,0,1,2) : Atom(0,0.75, 0.25),
        (0,3,2,1) : Atom(0,0,0),
        (1,0,3,1) : Atom(0,0.25,0.25),
        (2,1,0,3) : Atom(0,0.5,0),
        (3,2,1,0) : Atom(0,0.75, 0.25)
    } 


    @staticmethod
    def determine_hybridization(delta):
        sign_pattern = np.sign(delta)
        if np.array_equal(sign_pattern, [1, 1, 1]) or np.array_equal(sign_pattern, [-1, -1, -1]):       # Type a
            return 0
        elif np.array_equal(sign_pattern, [1, -1, -1]) or np.array_equal(sign_pattern, [-1, 1, 1]):   # Type b
            return 1
        elif np.array_equal(sign_pattern, [-1, 1, -1]) or np.array_equal(sign_pattern, [1, -1, 1]):   # Type c
            return 2
        elif np.array_equal(sign_pattern, [-1, -1, 1]) or np.array_equal(sign_pattern, [1, 1, -1]):   # Type d
            return 3
        else:
            raise Exception("error in hybridization")
    
    a = 0.5431e-9
    def classify_permutation(p: tuple) -> int | str:
        """
        Classifies a tuple if it's a cyclic permutation of (0,1,2,3) or (3,2,1,0).
        """
        if not isinstance(p, tuple) or sorted(list(p)) != [0, 1, 2, 3]:
            raise ValueError

        if all((p[i] + 1) % 4 == p[(i + 1) % 4] for i in range(4)):
            return 0
        if all((p[i] - 1) % 4 == p[(i + 1) % 4] for i in range(4)):
            
            return (3 - p[0] + 4) % 4 + 1
        
        # If the permutation is valid but doesn't fit the defined patterns.
        return "Error: Sequence is not cyclically ascending or descending."
    def __init__(self, channel_length_blocks : int, channel_width_blocks : int, channel_thickness_blocks : int, periodic = False, orientation = (0,1,2,3)):
        """
        Builds the nanowire unit cell. 
        
        """
        self.Nz = channel_thickness_blocks
        self.Nx = channel_length_blocks
        self.Ny = channel_width_blocks
        self.raw_deltas = {
            ((0 + SiliconUnitCell.classify_permutation(orientation)) % 2): [(+0.25, +0.25, +0.25), (+0.25, -0.25, -0.25),
                (-0.25, +0.25, -0.25), (-0.25, -0.25, +0.25)],
            ((1 + SiliconUnitCell.classify_permutation(orientation)) % 2): [(-0.25, -0.25, -0.25), (-0.25, +0.25, +0.25),
                (+0.25, -0.25, +0.25), (+0.25, +0.25, -0.25)]
        }
        self.ATOM_POSITIONS = [] # the order of atoms in this cell is very important as this governs the layer structure 
        self.neighbors = {}
        self.danglingBonds = {}
        self.addAtoms(SiliconUnitCell.orientation_to_start[orientation])
        self.map_neighbors_and_dangling_bonds()
        self.atoms_by_layer = self.split_into_layers()
        

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
            sublattice = SiliconUnitCell._sublattice(atom)
            self.neighbors[atom] = []
            self.danglingBonds[atom] = []
            for delta in self.raw_deltas[sublattice]:
                neighbor = atom.add(delta)
                # Ignore dangling bonds in X direction
                if not self.check_in_y_direction(neighbor) or not self.check_in_z_direction(neighbor):
                    self.danglingBonds[atom].append((neighbor, SiliconUnitCell.determine_hybridization(delta)))   
                elif self.check_in_x_direction(neighbor):
                    # Only add as neighbor if inside cell in all directions
                    if neighbor in atom_set:
                        l, m, n = SiliconUnitCell.directionalCosine(delta)
                        self.neighbors[atom].append((neighbor, delta, l, m, n))
                    
    def get_neighbors(self, atom: Atom):
        """Return a list of neighboring Atom objects using raw_deltas and sublattice."""
        sublattice = SiliconUnitCell._sublattice(atom)
        deltas = self.raw_deltas[sublattice]
        neighbors = [atom.add(delta) for delta in deltas]
        return neighbors
    
    def split_into_layers(self):
        """Group atoms into longitudinal layers based on x coordinate.

        Each increment of x by 0.25 corresponds to the next layer, so there are Nx*4 layers.
        """
        num_layers = int(self.Nx * 4)
        atoms_by_layer = [[] for _ in range(num_layers)]

        for atom in self.ATOM_POSITIONS:
            layer_num = int(atom.x * 4 + 1e-9)  
            if 0 <= layer_num < num_layers:
                atoms_by_layer[layer_num].append(atom)
            else:

                pass

        return atoms_by_layer
        
        