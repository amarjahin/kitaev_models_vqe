from networkx import Graph
from collections import Counter

class KitaevModel(Graph): 
    """A class holding information about the kitaev model on different lattices. 
       In general, term of the Hamiltonian is represented by a string. Example: 
       'IIIZ' is a tensor product operator acting on 4 qubit, with Z acting on the 
       0-th qubit and identity act on all others. 'IZII' mean Z act on the 2nd qubit.  

    """

    def __init__(self, L, J, H=(0,0,0), lattice_type='honeycomb_torus'): 
        """To initialize the Kitaev model you need to specify the size L, and the 
           strength of the interaction J

        Args:
            L (tuple): L = (Lx, Ly) such that Lx is the number of unit cells in the x direction 
                       and Ly is the number of unit cells in the y direction.
            J (tuple): J = (jx, jy, jz) is the strength of the interactions in the x, y, z links.
            lattice_type (str, optional): The different kinds on lattices you can choose from are:
                        'honeycomb_torus'
                        'honeycomb_open'
                        Defaults to 'honeycomb_torus'.
        """
        Graph.__init__(self)
        self.Lx, self.Ly = L[0], L[1]
        self.jx, self.jy, self.jz = J[0], J[1], J[2]
        self.J = J
        self.magnetic_field = H
        self.edge_dict = {'X':1, 'Y':2, 'Z':3}
        self.lattice_type = lattice_type
        self.number_of_unit_cells = self.Lx*self.Ly
        lattice_to_func = {'honeycomb_torus':self.honeycomb_torus, 'honeycomb_open':self.honeycomb_open,
                            'eight_spins_4_8_8':self.eight_spins_4_8_8}
        # define_lattice = lattice_to_func.get(lattice_type)
        define_lattice = lattice_to_func[lattice_type]
        define_lattice(L,J) 

        # add external magnetic field terms to the spin Hamiltonian if they exist 
        if self.magnetic_field != (0,0,0): 
            for i in range(self.number_of_spins): 
                for d in ['X', 'Y', 'Z']: 
                    term = ['I' for _ in range(self.number_of_spins)]
                    term[i] = d
                    k = self.edge_dict[d] - 1
                    self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term[::-1]), mag=-1*self.magnetic_field[k])



    def unit_cell_indx(self, i, j):
        """This gives a 1D parametrization for a 2D system. 
        This is done in a zig-zag kind of way. Example 3x3:
                    6 7 8
                    5 4 3 
                    0 1 2 

        Args:
            i (int): The 'x' coordinate
            j (int): The 'y' coordinate

        Returns:
            int: The 1D parametrization
        """
        i = i % self.Lx
        j = j % self.Ly 
        indx = (-1)**j * i + self.Lx * j + (self.Lx-1) * (1 - (-1)**j)/2
        return int(indx)


    def add_term_to_hamiltonian(self, h, term, mag): 
        """This do the following h = h + mag*term.

        Args:
            h (dict): A dictionary holding terms in the Hamiltonian as keys, and magnitudes as values.
            term (str): A string for the term to be added, eg. 'IIIZ'.
            mag (float): The magnitude or coefficient of the term

        Returns:
            dict: updated Hamiltonian h = h + mag*term
        """
        c = Counter(h)
        c.update(Counter({term:mag}))
        return dict(c)


    def honeycomb_torus(self, L, J): 
        self.number_of_spins = self.number_of_unit_cells*2
        self.number_of_Dfermions = self.number_of_spins*2
        self.number_of_Dfermions_u = self.number_of_spins//2
        self.spin_hamiltonian = {}
        self.fermionic_hamiltonian = {}
        for j in range(self.Ly): 
            for i in range(self.Lx): 
                # create edges of the graph
                cell_indx = self.unit_cell_indx(i, j)
                node_a_indx = 2*cell_indx
                node_b_indx = 2*cell_indx + 1
                self.add_edges_from([(node_a_indx, node_b_indx, {'weight':self.jz, 'label':'Z'})])
                URN_cell_indx = self.unit_cell_indx(i + 1, j)
                URN_node_a_inx = 2 * URN_cell_indx
                ULN_cell_indx = self.unit_cell_indx(i, j + 1)
                ULN_node_a_inx = 2 * ULN_cell_indx
                self.add_edges_from([(URN_node_a_inx, node_b_indx, {'weight':self.jx, 'label':'X'})])
                self.add_edges_from([(ULN_node_a_inx, node_b_indx, {'weight':self.jy, 'label':'Y'})])

                # create the original spin Hamiltonian terms
                term_z = ['I' for _ in range(self.number_of_spins)]
                term_x = ['I' for _ in range(self.number_of_spins)]
                term_y = ['I' for _ in range(self.number_of_spins)]
                term_z[node_a_indx],    term_z[node_b_indx] = 'Z', 'Z'
                term_x[URN_node_a_inx], term_x[node_b_indx] = 'X', 'X'
                term_y[ULN_node_a_inx], term_y[node_b_indx] = 'Y', 'Y'
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_z[::-1]), mag=-1*self.jz)
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_x[::-1]), mag=-1*self.jx)
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_y[::-1]), mag=-1*self.jy)

    def honeycomb_open(self, L, J): 
        self.number_of_spins = self.number_of_unit_cells*2
        self.number_of_Dfermions = self.number_of_spins*2
        self.number_of_Dfermions_u = self.number_of_spins//2
        self.spin_hamiltonian = {}
        # self.fermionic_hamiltonian = {}
        for j in range(self.Ly): 
            for i in range(self.Lx): 
                # create edges of the graph
                cell_indx = self.unit_cell_indx(i, j)
                node_a_indx = 2*cell_indx
                node_b_indx = 2*cell_indx + 1
                self.add_edges_from([(node_a_indx, node_b_indx, {'weight':self.jz, 'label':'Z'})])
                # create the original spin Hamiltonian term 
                term_z = ['I' for _ in range(self.number_of_spins)]
                term_z[node_a_indx],    term_z[node_b_indx] = 'Z', 'Z'
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_z[::-1]), mag=-1*self.jz)
                
                # These conditions make sure the conditions are open. 
                # If going to the neighboring sites are cycle back to the beginning 
                # then don't add these terms 
                URN_cell_indx = self.unit_cell_indx(i + 1, j)
                URN_node_a_inx = 2 * URN_cell_indx
                if (i+1) % self.Lx !=0:
                    self.add_edges_from([(URN_node_a_inx, node_b_indx, {'weight':self.jx, 'label':'X'})])
                    # create the original spin Hamiltonian term 
                    term_x = ['I' for _ in range(self.number_of_spins)]
                    term_x[URN_node_a_inx], term_x[node_b_indx] = 'X', 'X'
                    self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                                term=''.join(term_x[::-1]), mag=-1*self.jx)
                else: 
                    self.add_edges_from([(URN_node_a_inx, node_b_indx, {'weight':0, 'label':'X'})])

                ULN_cell_indx = self.unit_cell_indx(i, j + 1)
                ULN_node_a_inx = 2 * ULN_cell_indx
                if (j+1) % self.Ly !=0:
                    self.add_edges_from([(ULN_node_a_inx, node_b_indx, {'weight':self.jy, 'label':'Y'})])
                    # create the original spin Hamiltonian term 
                    term_y = ['I' for _ in range(self.number_of_spins)]
                    term_y[ULN_node_a_inx], term_y[node_b_indx] = 'Y', 'Y'
                    self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                                term=''.join(term_y[::-1]), mag=-1*self.jy)
                else: 
                    self.add_edges_from([(ULN_node_a_inx, node_b_indx, {'weight':0, 'label':'Y'})])


    def eight_spins_4_8_8(self, L, J): 
        self.number_of_spins = 8
        self.number_of_Dfermions = 16
        self.number_of_Dfermions_u = 4
        self.spin_hamiltonian = {}
        self.fermionic_hamiltonian = {}
        self.add_edges_from([(0, 1, {'weight':self.jz, 'label':'Z'})])
        self.add_edges_from([(2, 3, {'weight':self.jz, 'label':'Z'})])
        self.add_edges_from([(4, 5, {'weight':self.jz, 'label':'Z'})])
        self.add_edges_from([(6, 7, {'weight':self.jz, 'label':'Z'})])

        self.add_edges_from([(2, 1, {'weight':self.jx, 'label':'X'})])
        self.add_edges_from([(6, 5, {'weight':self.jx, 'label':'X'})])
        self.add_edges_from([(2, 5, {'weight':self.jy, 'label':'Y'})])
        self.add_edges_from([(6, 1, {'weight':self.jy, 'label':'Y'})])

        for e in self.edges: 
            term = ['I' for _ in range(self.number_of_spins)]
            term[e[0]],    term[e[1]] = self.edges[e]['label'], self.edges[e]['label']
            mag = self.J[self.edge_dict[self.edges[e]['label']] - 1]
            self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term[::-1]), mag=-1*mag)
                


    
    def four_eight_eight_torus(self): 
        self.number_of_spins = self.number_of_unit_cells*2
        self.number_of_Dfermions = self.number_of_spins*2
        self.number_of_Dfermions_u = self.number_of_spins//2
        self.spin_hamiltonian = {}
        None



    def jw_hamiltonian_u(self, u):
        """Get the Hamiltonian of the system with a fixed gauge transformed using Jordan-Wigner 

        Args:
            u (ndarray): A matrix holding the gauge information such that u[i,j] is the gauge value in the (i,j) link. 

        Returns:
            dict: A dictionary of the Hamiltonian, eg {'IIIZ':0.5,'IIZI:-1} means h = 0.5 * IIIZ - IIZI.
        """

        h = {}
        for e in self.edges: 
            if e[0] % 2 == 0: 
                i = e[0]
                j = e[1]
            else: 
                i = e[1]
                j = e[0]

            ip = i//2
            jp = j//2 

            mag = self.edges[e]['weight'] * u[i,j]
            term = ['I' for _ in range(self.number_of_Dfermions_u)]
            if ip == jp: 
                # h = self.add_term_to_hamiltonian(h=h, term=''.join(term[::-1]), mag=-1*mag)
                term[ip] = 'Z'
                h = self.add_term_to_hamiltonian(h=h, term=''.join(term[::-1]), mag=-1*mag)
            elif ip>jp: 
                term[ip] = 'X'
                term[jp] = 'X'
                for k in range(min(ip,jp)+1, max(ip,jp)):
                    term[k] = 'Z'
                h = self.add_term_to_hamiltonian(h=h, term=''.join(term[::-1]), mag=mag)
            else: 
                term[ip] = 'Y'
                term[jp] = 'Y'
                for k in range(min(ip,jp)+1, max(ip,jp)):
                    term[k] = 'Z'
                h = self.add_term_to_hamiltonian(h=h, term=''.join(term[::-1]), mag=mag)
                
        return h


    def jw_hamiltonian(self):
        h = {}
        # edge_dict = {'X':1, 'Y':2, 'Z':3}
        for e in self.edges: 
            if e[0] % 2 == 0: 
                i = e[0]
                j = e[1]
            else: 
                i = e[1]
                j = e[0]

            ip = i//2
            jp = j//2 

            edges_indx = 3*(ip + 1) + self.edge_dict[self.edges[e]['label']]

            mag = self.edges[e]['weight'] 
            term = ['I' for _ in range(self.number_of_Dfermions)]
            if ip == jp: 
                term[ip] = 'Z'
                term[edges_indx] = 'Z'
                h = self.add_term_to_hamiltonian(h=h, term=''.join(term[::-1]), mag=mag)
            elif ip>jp: 
                term[ip] = 'X'
                term[jp] = 'X'
                for k in range(min(ip,jp)+1, max(ip,jp)):
                    term[k] = 'Z'
                term[edges_indx] = 'Z'
                h = self.add_term_to_hamiltonian(h=h, term=''.join(term[::-1]), mag=-1*mag)
            else: 
                term[ip] = 'Y'
                term[jp] = 'Y'
                for k in range(min(ip,jp)+1, max(ip,jp)):
                    term[k] = 'Z'
                term[edges_indx] = 'Z'
                h = self.add_term_to_hamiltonian(h=h, term=''.join(term[::-1]), mag=-1*mag)
            
            # add magnetic field terms if they exist  
            if self.magnetic_field != (0,0,0): 
                term_1 = ['I' for _ in range(self.number_of_Dfermions)]
                term_2 = ['I' for _ in range(self.number_of_Dfermions)]
                term_1[ip] = 'Y'
                term_1[edges_indx] = 'X'
                for k in range(min(ip,edges_indx)+1, max(ip,edges_indx)):
                    term_1[k] = 'Z'
                mag = self.magnetic_field[self.edge_dict[self.edges[e]['label']] - 1]
                h = self.add_term_to_hamiltonian(h=h, term=''.join(term_1[::-1]), mag=-1*mag)

                term_2[jp] = 'X'
                term_2[edges_indx] = 'Y'
                for k in range(min(jp,edges_indx)+1, max(jp,edges_indx)):
                    term_2[k] = 'Z'
                h = self.add_term_to_hamiltonian(h=h, term=''.join(term_2[::-1]), mag=mag)
                
        
        return h