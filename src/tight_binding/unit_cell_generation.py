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
    @staticmethod
    def createRotationMatrix(theta):
        return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    
    raw_deltas = {
            0: [(+0.25, +0.25, +0.25), (+0.25, -0.25, -0.25),
                (-0.25, +0.25, -0.25), (-0.25, -0.25, +0.25)],
            1: [(-0.25, -0.25, -0.25), (-0.25, +0.25, +0.25),
                (+0.25, -0.25, +0.25), (+0.25, +0.25, -0.25)]
        }
    
    @staticmethod
    def baseCoordinates(n=1):
        "returns nth layer base coordinates" 
        if n == 0:
            raise Exception("should be positive")
        return [Atom(0.5,0.5,0 + n - 1), Atom(.25,.25,.25 + n - 1), Atom(0,.5,.5 + n -1), Atom(.25,.75,.75 + n - 1), Atom(.5,.5,1 + n - 1)]
    
    def __init__(self,  N= 1, cleavage_plane = (0,0,1)):
        self.HKL = cleavage_plane
        self.N = N
        self.R = self.change_of_base_matrix() if self.HKL != (0, 0, 1) else np.eye(3)
        self.ATOM_POSITIONS = []
        self.Z_ATOMS = {}
        self.createBasis()
        self.neighbors, self.danglingBonds = self.mapNeighbors()
        self.Nx, self.Ny, self.Nz = 0,0,0

    def createBasis(self):
        for layer in range(1, self.N + 1):
            base_coordinates = UnitCell.baseCoordinates(layer)
            for atom in base_coordinates:
                if atom not in self.ATOM_POSITIONS:
                    self.ATOM_POSITIONS.append(atom)
                    self.Z_ATOMS.update({atom.z : atom})
            

    
    def change_of_base_matrix(self):
        new_z = self.normalize(self.HKL)
        global_z = np.array([0, 0, 1])
        if np.allclose(new_z, global_z):
            return np.eye(3)
        new_y = self.normalize(np.cross(global_z, new_z))
        new_x = self.normalize(np.cross(new_y, new_z))
        return np.column_stack([new_x, new_y, new_z])
    
    
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
                    l,m,n = UnitCell.directionalCosine(delta)

                    mappedNeighbor = self.Z_ATOMS[neighbor.z]
                    #print(f"{atom} neighbor is {neighbor} but its mapped to {mappedNeighbor}")
                    neighborsMap[atom].append((mappedNeighbor, delta, l,m,n))
                        
                else:
                    danglingBonds[atom].append((neighbor, UnitCell.determine_hybridization(delta)))
        
        return neighborsMap, danglingBonds
    
    def printNeighborMap(self):
        for atom in self.neighbors:
            neighbors = [(delta, neighbor) for neighbor, delta, l, m, n in self.neighbors[atom]]
            
            print(f"the atom is {atom} and the neighbors are: {neighbors}")
    
    
    
    def setGrid(self, Nx, Ny, Nz):
        # so corners are (.5,.5) , (.25,.25), (0,.5), (0.25,.75)
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
     
    
    def cellToGrid(self, r):
        x,y,z = r
        arr = np.array([x,y])
        rot = UnitCell.createRotationMatrix(-np.pi/4 - np.pi/2)
        arr = rot @ arr
        arr -= np.array([0.25,0.25])
        
        gx,gy, gz = arr[0] / 0.25 * (self.Nx- 1), arr[1] / 0.25 * (self.Ny- 1), z / self.N * (self.Nz - 1) 
        return gx,gy,gz
    def gridToCell(self, r):
        gx, gy, gz = r
        arr = np.array([
            gx * 0.25 / (self.Nx - 1) + 0.25,
            gy * 0.25 / (self.Ny - 1) + 0.25
        ])
        rot = UnitCell.createRotationMatrix(-np.pi/4 - np.pi/2)
        x, y = rot.T @ arr        
        z = gz * self.N / (self.Nz - 1)
        return x,y,z
        
                    
                