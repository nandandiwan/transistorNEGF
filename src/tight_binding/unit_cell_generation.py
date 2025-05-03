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
    

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __repr__(self):
        return f"Atom({self.x}, {self.y}, {self.z})"
    

class UnitCellGeneration:
    _DELTAS = {                       
        0: [(+0.25,+0.25,+0.25), (+0.25,-0.25,-0.25),
            (-0.25,+0.25,-0.25), (-0.25,-0.25,+0.25)],   # type-0 sub-lattice
        1: [(-0.25,-0.25,-0.25), (-0.25,+0.25,+0.25),
            (+0.25,-0.25,+0.25), (+0.25,+0.25,-0.25)]    # type-1 sub-lattice
    }
    
    
    @staticmethod
    def _delta(a, b):   
        return (b.x - a.x, b.y - a.y, b.z - a.z)

    @staticmethod
    def _sublattice(atom):    
        return (round(atom.x*4) + round(atom.y*4) + round(atom.z*4)) & 1
    

    def __init__(self, N):
        #print(N)
        self.atoms = {}      # basic graph design 
        self.visited = set() # dfs
        self.N = N
        # base node 
        self.addAtoms(Atom(0, 0, 0), 0)
        self.periodicityFix = {}
        self.atoms_nonPeriodic = self.atoms.copy()
        
        self.hydrogens = {}
        self.numHydrogens = 0

        for atom in self.atoms:
            
            listofatoms = self.atoms[atom]
            for i in listofatoms:
              
                value = i
                if not self.checkIfAllowed(value):
                    # we need to update the periodic neighbor
                    delta = (value.x - atom.x, value.y - atom.y, value.z - atom.z)
                    try:
                        newNeighbor = Atom((atom.x + delta[0]) % 1, (atom.y + delta[1]) % 1, (atom.z + delta[2]))
                        self.atoms[newNeighbor].append(atom)
                       
                    except:
                        print(atom, value, newNeighbor)
                        return 
                    
            

            self.atoms[atom] = list(filter(self.checkIfAllowed, self.atoms[atom]))
            
            self.hydrogen_passification()
            


    def checkIfAllowed(self, newAtom):
        # check if atoms are in proper cell 
        return not (newAtom.x < 0 or newAtom.y < 0 or newAtom.z < 0 or newAtom.z >= self.N or newAtom.x >= 1 or newAtom.y >= 1)
        
    def checkIfAllowedInZDirection(self, newAtom):
        return not (newAtom.z < 0 or newAtom.z >= self.N)

    def addAtoms(self, base, atomType):
        # dfs 
        if base in self.visited:
            return
        self.visited.add(base)

        if atomType == 0:
            atom1 = Atom(base.x + 0.25, base.y + 0.25, base.z + 0.25)
            atom2 = Atom(base.x + 0.25, base.y - 0.25, base.z - 0.25)
            atom3 = Atom(base.x - 0.25, base.y + 0.25, base.z - 0.25)
            atom4 = Atom(base.x - 0.25, base.y - 0.25, base.z + 0.25)
        elif atomType == 1:
            atom1 = Atom(base.x - 0.25, base.y - 0.25, base.z - 0.25)
            atom2 = Atom(base.x - 0.25, base.y + 0.25, base.z + 0.25)
            atom3 = Atom(base.x + 0.25, base.y - 0.25, base.z + 0.25)
            atom4 = Atom(base.x + 0.25, base.y + 0.25, base.z - 0.25)
        else:
            return

        newAtoms = [atom1, atom2, atom3, atom4]
        
        for newAtom in newAtoms:
            
            # If base has no neighbors yet, initialize its neighbor list.
            if self.checkIfAllowedInZDirection(newAtom):
                if base not in self.atoms:
                    self.atoms[base] = [newAtom]
                elif newAtom not in self.atoms[base]:
                    self.atoms[base].append(newAtom)
            
            # Recursively add neighbors (DFS)
            if self.checkIfAllowed(newAtom):
                self.addAtoms(newAtom, (atomType + 1) % 2)
                


    def neighborTable(self): # finds the directional cosines: 
        neighborInformation = {}
        for atom in self.atoms:
            neighborInformation[atom] = {}
            for i in range(len(self.atoms[atom])):
                nonPeriodicNeighbor = self.atoms_nonPeriodic[atom][i]
     
                # Compute the difference vector between neighbor and atom.
                dx = nonPeriodicNeighbor.x - atom.x
                dy = nonPeriodicNeighbor.y - atom.y
                dz = nonPeriodicNeighbor.z - atom.z
                
                delta = (dx, dy, dz)
                newNeighbor = Atom((atom.x + delta[0]) % 1, (atom.y + delta[1]) % 1, (atom.z + delta[2]))
                norm = np.sqrt(dx**2 + dy**2 + dz**2)
                
                # Calculate the directional cosines: l, m, n.
                if norm != 0:
                    l = dx / norm
                    m = dy / norm
                    n = dz / norm
                else:
                    l, m, n = 0.0, 0.0, 0.0
                
                neighborInformation[atom].update({newNeighbor: (delta, l, m, n)})
        
        return neighborInformation
    
    def hydrogen_passification(self):
        # we need to give each hydrogen an index 
        hydrogenIndex = 0
        for atom in self.atoms:
            missing = self.dangling_bonds(atom, only_z=True)
            for hydrogen_ in missing:
                #print(f"silicon atom is: {atom} hydrogen atom is {hydrogen_}")
                
                hydrogen = Atom(hydrogen_[0] + atom.x, hydrogen_[1]+ atom.y, hydrogen_[2] + + atom.z)
                dx,dy,dz = hydrogen_
                norm = np.sqrt(dx**2 + dy**2 + dz**2)
                l = dx / norm
                m = dy / norm
                n = dz / norm
                self.hydrogens[hydrogen] = [atom, hydrogen_, hydrogenIndex, l,m,n]
                hydrogenIndex += 1
                
                
    
    def dangling_bonds(self, atom, only_z= False):
        """
        Return the list of bond-direction vectors that *should* exist for
        `atom` but do not (because the neighbour lies outside the slab).
        """
        missing = []
        for dx, dy, dz in self._DELTAS[self._sublattice(atom)]:
            nx, ny, nz = atom.x + dx, atom.y + dy, atom.z + dz

            # check if it is outside the cell in z direction 
            if nz < 0 or nz >= self.N:
                missing.append((dx, dy, dz))
                continue
            if only_z == False:
                n_atom = Atom(nx % 1, ny % 1, nz)   # wrap in x, y (periodic)
                if n_atom not in self.atoms:        
                    missing.append((dx, dy, dz))

        return missing     
    
    def create_linear_potential(self, V):
        linear_potential = lambda i, V : i / (self.N) * V
        #print(self.N)
        potential = np.array([linear_potential(i, V) for i in range(self.N * 4 + 1)])
        #print(potential)
        return potential
    # OLD
    
    def determine_hybridization(signs):
        # Extract just the signs
        sign_pattern = np.sign(signs)
        
        # Ensure first sign (s orbital) is positive for consistent comparison
        if sign_pattern[0] < 0:
            sign_pattern = -sign_pattern
        
        # Map each sign pattern to its hybridization index
        if np.array_equal(sign_pattern, [1, 1, 1, 1]):       # Type a
            return 0
        elif np.array_equal(sign_pattern, [1, 1, -1, -1]):   # Type b
            return 1
        elif np.array_equal(sign_pattern, [1, -1, 1, -1]):   # Type c
            return 2
        elif np.array_equal(sign_pattern, [1, -1, -1, 1]):   # Type d
            return 3
        
        return None 
        
        
    def __str__(self):
        s = ""
        for atom, neighbors in self.atoms.items():
            s += f"{atom} -> {neighbors}" + "\n"
            
        return s
    
