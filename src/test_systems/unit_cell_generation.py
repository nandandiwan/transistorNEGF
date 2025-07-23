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
    
class GrapeheneZigZagCell:
    a = 1
    sin60 = np.sqrt(3) / 2
    cos60 = 1/2 
    base = [Atom(sin60,0,0), Atom(0, cos60, 0), Atom(0,1.5,0), Atom(sin60, 2,0)]
    
    deltas = {
        0 : [(0,1,0), (-sin60, -cos60, 0),(sin60, -cos60, 0)],
        1 : [(0,-1,0), (-sin60, cos60, 0),(sin60, cos60, 0)]
    }
    def __init__(self, num_layers_x, num_layers_y):
        self.num_layers_y = num_layers_y
        self.num_layers_x = num_layers_x
        
        self.max_X = self.sin60 + num_layers_x *2*self.sin60
        self.max_Y = 3 *num_layers_y - 1
         
        
        self.layer = self.create_first_layer()
        self.structure = self.create_full_structure()
        
        self.sublattice = {}
        
        for idx, atom in enumerate(self.structure):
            self.sublattice[atom] = (idx+1)%2
        
        self.neighbors, self.dangling = self.neighbors_and_dangling()
    def create_first_layer(self):
        layer = []
        for y in range(self.num_layers_y):
            delta = (0,3*y, 0)
            new_atoms = map(lambda x: x.add(delta), GrapeheneZigZagCell.base)
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
            deltas = GrapeheneZigZagCell.deltas[self.get_sublattice(atom)]
            for delta in deltas:
                neighbor = atom.add(delta)
                
                if (not self.check_in_y_direction(neighbor)):
                    l, m, n = GrapeheneZigZagCell.directionalCosine(delta)
                    dangling_bonds[atom].append((neighbor, delta, l, m, n))  
                elif (self.check_in_y_direction(neighbor) and self.check_in_x_direction(neighbor)):

                    # in the structure 
                    l, m, n = GrapeheneZigZagCell.directionalCosine(delta)
                    neighbors[atom].append((neighbor, delta, l, m, n))   
                # ignore x 
        
        return neighbors, dangling_bonds
