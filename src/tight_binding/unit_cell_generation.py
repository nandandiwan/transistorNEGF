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
    
    a = 0.5431
    @staticmethod
    def baseCoordinates(n=1):
        "returns nth layer base coordinates" 
        if n == 0:
            raise Exception("should be positive")
        return [Atom(0.5,0.5,0 + n - 1), Atom(.25,.25,.25 + n - 1), Atom(0,.5,.5 + n -1), Atom(.25,.75,.75 + n - 1), Atom(.5,.5,1 + n - 1)]
    
    @staticmethod
    def nextDelta():
        deltas = [(-0.25, -0.25, 0.25),
                (-0.25,  0.25, 0.25),
                ( 0.25,  0.25, 0.25),
                ( 0.25, -0.25, 0.25)]
        while True:
            yield from deltas                
            yield from UnitCell.nextDelta() 
    
    def __init__(self,  N= 1, thickness = None, cleavage_plane = (0,0,1)):
        self.HKL = cleavage_plane
        
        self.R = self.change_of_base_matrix() if self.HKL != (0, 0, 1) else np.eye(3)
        self.ATOM_POSITIONS = []
        self.ATOM_POTENTIAL = {}
        self.OLD_POTENTIAL = None
        self.Z_ATOMS = {}
        
        if thickness is None:
            self.N = N
            self.createBasis()
        else:
            self.thickness = thickness
            self.layers = int((thickness / UnitCell.a / .25 + 1)) 
            print(self.layers)
            if self.layers < 2:
                raise RuntimeError("Thickness must be enough for two layers (0.5431 nm)")       
            self.fillLayers()
            
        
        self.neighbors, self.danglingBonds = self.mapNeighbors()
        
            
            
        self.voltageProfile = None
        self.oldVoltageProfile = None
        self.setVoltage()
        self.Nx, self.Ny, self.Nz = 0,0,0
        

    def createBasis(self):
        for layer in range(1, self.N + 1):
            base_coordinates = UnitCell.baseCoordinates(layer)
            for atom in base_coordinates:
                if atom not in self.ATOM_POSITIONS:
                    self.ATOM_POSITIONS.append(atom)
                    self.Z_ATOMS.update({atom.z : atom})
            
    def fillLayers(self):
        base_atom = Atom(0.5, 0.5, 0.0)
        self.ATOM_POSITIONS.append(base_atom)
        self.Z_ATOMS[0.0] = base_atom

        # persistent generator that cycles through the four Δ‑vectors
        delta_gen = UnitCell.nextDelta()      # one generator, reused

        for layer in range(1, self.layers):
            layer_z = layer * 0.25
            previous_atom = self.Z_ATOMS[layer_z - 0.25]

            delta = next(delta_gen)           # <- actual tuple

            atom = previous_atom.add(delta)
            self.ATOM_POSITIONS.append(atom)
            self.Z_ATOMS[atom.z] = atom
                
    
    def setVoltage(self, voltage=None):
        for atom in self.ATOM_POSITIONS:
            if voltage is not None:
                self.voltageProfile = voltage
                i,j,k = self.xyz_to_grid(atom.x,atom.y,atom.z)
            
                self.ATOM_POTENTIAL[atom] = voltage[int(i),int(j),int(k)]
        
            
            else:
                self.ATOM_POTENTIAL[atom] = 0
            
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
        try:
            return not (newAtom.x < 0 or newAtom.y < 0 or newAtom.z < 0 or newAtom.z >= self.N or newAtom.x >= 1 or newAtom.y >= 1)
        except:
            return not (newAtom.x < 0 or newAtom.y < 0 or newAtom.z < 0 or newAtom.z >= (self.layers - 1)  * 0.25 or newAtom.x >= 1 or newAtom.y >= 1)
        
    def checkIfAllowedInZDirection(self, newAtom):
        try:
            return not (newAtom.z < 0 or newAtom.z > self.N)
        except:
            return not (newAtom.z < 0 or newAtom.z > (self.layers - 1)  * 0.25)
    
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
     
        
    def xyz_to_grid(self, x, y, z):
        # shift
        x,y = x - 0.25, y - 0.5
        rot = UnitCell.createRotationMatrix(-3 * np.pi / 4)
        x,y = rot @ np.asarray([x,y])
        x,y = x + np.sqrt(2)/8, y+ np.sqrt(2)/8
        
        #scale
        x,y = 4 * x / np.sqrt(2), 4 * y / np.sqrt(2)
        x,y = np.round(x), np.round(y)
        x,y = x * (self.Nx - 1), y * (self.Ny - 1)
        
        # N is number of unit cells ie max height is N, 
        try:
            z = np.round(z / self.N * (self.Nz - 1))
        except:
            z = np.round(z / ((self.layers - 1) * 0.25) * (self.Nz - 1))
        
        return x, y,z
    def gridToCell(self, r):
        """ outdated fix this with method in selfConsistent.ipynb"""
        gx, gy, gz = r
        arr = np.array([
            gx * 0.25 / (self.Nx - 1) + 0.25,
            gy * 0.25 / (self.Ny - 1) + 0.25
        ])
        rot = UnitCell.createRotationMatrix(-np.pi/4 - np.pi/2)
        x, y = rot.T @ arr        
        z = gz * self.N / (self.Nz - 1)
        return x,y,z
        
                    
                