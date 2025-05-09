import numpy as np

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
    
    raw_deltas = {
            0: [(+0.25, +0.25, +0.25), (+0.25, -0.25, -0.25),
                (-0.25, +0.25, -0.25), (-0.25, -0.25, +0.25)],
            1: [(-0.25, -0.25, -0.25), (-0.25, +0.25, +0.25),
                (+0.25, -0.25, +0.25), (+0.25, +0.25, -0.25)]
        }
    
    def __init__(self, N= 1):
  
        self.N = N

        self.ATOM_POSITIONS = [Atom(-.5,.5,0)]
        self.Z_ATOMS = {0: Atom(-.5,.5,0)}
        self.createBasis(0.25,(-0.25,-0.25,0.25))
        self.neighbors, self.danglingBonds = self.mapNeighbors()

    
    def createBasis(self, z, shift):
        shift1 = (-0.25,-0.25,0.25)
        shift2 = (0.25,-.25,0.25)

        
        if z <= self.N:
            newAtom = self.ATOM_POSITIONS[-1].add(shift)
            self.ATOM_POSITIONS.append(newAtom)
            self.Z_ATOMS.update({z : newAtom})
            if shift == shift1:
                self.createBasis(z + .25, shift2)
            else:
                self.createBasis(z + .25, shift1)
    

    
    
    def checkIfAllowed(self, newAtom):
        # check if atoms are in proper cell 
        return not (newAtom.x < 0 or newAtom.y < 0 or newAtom.z < 0 or newAtom.z >= self.N or newAtom.x >= 1 or newAtom.y >= 1)
        
    def checkIfAllowedInZDirection(self, newAtom):
        return not (newAtom.z < 0 or newAtom.z > self.N)
    
    def determine_hybridization(delta):
        # Extract just the signs
        sign_pattern = np.sign(delta)
        
        # Map each sign pattern to its hybridization index
        if np.array_equal(sign_pattern, [1, 1, 1]):       # Type a
            return 0
        elif np.array_equal(sign_pattern, [1, -1, -1]):   # Type b
            return 1
        elif np.array_equal(sign_pattern, [-1, 1, -1]):   # Type c
            return 2
        elif np.array_equal(sign_pattern, [-1, -1, 1]):   # Type d
            return 3
        else:
            Exception("implement this dumbass")
    def mapNeighbors(self):
        """This gives the list of neighbors of an atom
        For example say i am working with atom at 0.25,.25,.25. I find the sublattice and then the neighbors.
        Each neighbors corresponds to an atom in the TB Hamiltonian - via periodicity or direct neighbor
        
        """
        atoms = self.ATOM_POSITIONS
        neighborsMap = {}
        danglingBonds = {}
        for atom in atoms:
            sublattice = UnitCell._sublattice(atom)
            neighborsMap[atom] = []
            danglingBonds[atom] = []
            for delta in UnitCell.raw_deltas[sublattice]:
                neighbor = atom.add(tuple(delta))
                if self.checkIfAllowedInZDirection(neighbor):
                    mappedNeighbor = self.Z_ATOMS[neighbor.z]
                    l,m,n = UnitCell.directionalCosine(delta)
                    neighborsMap[atom].append((mappedNeighbor, delta, l,m,n))
                else:
                    danglingBonds[atom].append( (neighbor, UnitCell.determine_hybridization(delta)))
        
        return neighborsMap, danglingBonds
    
    def printNeighborMap(self):
        for atom in self.neighbors:
            neighbors = [(delta, neighbor) for neighbor, delta, l, m, n in self.neighbors[atom]]
            
            print(f"the atom is {atom} and the neighbors are: {neighbors}")
                    
                