from networkx import Graph
from numpy import zeros
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
                            'eight_spins_4_8_8':self.eight_spins_4_8_8, 'square_octagon_torus':self.square_octagon_torus, 
                            'square_octagon_open':self.square_octagon_open}
        # edge_direction_dict = {'honeycomb_torus':self.edge_direction_honeycomb, 'honeycomb_open':self.edge_direction_honeycomb,
        #                     'eight_spins_4_8_8':self.edge_direction_square_octagon,
        #                     'square_octagon_torus':self.edge_direction_square_octagon,
        #                     'square_octagon_open':self.edge_direction_square_octagon}
        # self.edge_direction = edge_direction_dict[self.lattice_type]
        define_lattice = lattice_to_func[lattice_type]
        define_lattice() 
        self.number_of_Dfermions = self.number_of_spins*2
        self.number_of_Dfermions_u = self.number_of_spins//2
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


    def edge_direction_honeycomb(self,e): 
        if e[0] % 2 == 0: 
            i = e[0]
            j = e[1]
        else: 
            i = e[1]
            j = e[0]

        return i, j

    def site_qubit_label_honeycomb(self, i): 
        return i//2

    def edge_qubit_label_honeycomb(self,e): 
        i = self.edge_direction_honeycomb(e)[0]
        ip = self.site_qubit_label_honeycomb(i)
        return 3*(ip) + self.edge_dict[self.edges[e]['label']] + self.number_of_Dfermions_u - 1

    def edge_direction_square_octagon(self,e): 
        r0, r1 =  e[0] % 4, e[1] % 4
        if r0 < r1:  
            i = e[0]
            j = e[1]
        else: 
            i = e[1]
            j = e[0]

        return i, j


    def site_qubit_label_square_octagon(self, i): 
        return i//2

    def edge_qubit_label_square_octagon(self,e): 
        i,j = self.edge_direction_square_octagon(e)
        ip = self.site_qubit_label_square_octagon(i)
        if self.edges[e]['label'] == 'X': 
            return 6*(ip//2) + 2*(i % 4) + self.edge_dict['X'] + self.number_of_Dfermions_u -1
        if self.edges[e]['label'] == 'Y': 
            return 6*(ip//2) + 2*(ip % 2) + self.edge_dict['Y'] + self.number_of_Dfermions_u -1
        if self.edges[e]['label'] == 'Z': 
            if i//4 == j//4: # this is to take care of the cases where the boundary condition tie the unit cell with itsself
                return 6*(ip//2) + (j % 4) + self.edge_dict['Z'] + self.number_of_Dfermions_u - 1 
            else: 
                return 6*(ip//2) + (i % 4) + self.edge_dict['Z'] + self.number_of_Dfermions_u - 1 

            # if self.Lx != 1 and self.Ly != 1: 
            #     return 6*(ip//2) + (i % 4) + self.edge_dict['Z'] + self.number_of_Dfermions_u - 1 
            # elif self.Lx == 1 and self.Ly == 1: 
            #     return 6*(ip//2) + (j % 4) + self.edge_dict['Z'] + self.number_of_Dfermions_u - 1 
            # elif self.Lx != 1 and self.Ly == 1:
            #     if (i%4) =    

    def honeycomb_torus(self): 
        self.edge_direction = self.edge_direction_honeycomb
        self.site_qubit_label = self.site_qubit_label_honeycomb
        self.edge_qubit_label = self.edge_qubit_label_honeycomb
        self.number_of_spins = self.number_of_unit_cells*2
        self.spin_hamiltonian = {}
        # self.fermionic_hamiltonian = {}
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
        return None

    def honeycomb_open(self): 
        self.edge_direction = self.edge_direction_honeycomb
        self.site_qubit_label = self.site_qubit_label_honeycomb
        self.edge_qubit_label = self.edge_qubit_label_honeycomb
        self.number_of_spins = self.number_of_unit_cells*2
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
                elif i != 0: 
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
                elif j != 0: 
                    self.add_edges_from([(ULN_node_a_inx, node_b_indx, {'weight':0, 'label':'Y'})])
        return None

    def eight_spins_4_8_8(self): 
        self.edge_direction = self.edge_direction_square_octagon
        self.site_qubit_label = self.site_qubit_label_square_octagon
        self.edge_qubit_label = self.edge_qubit_label_square_octagon
        self.number_of_spins = 8
        self.spin_hamiltonian = {}
        self.fermionic_hamiltonian = {}
        self.add_edges_from([(6, 0, {'weight':self.jz, 'label':'Z'})])
        self.add_edges_from([(2, 4, {'weight':self.jz, 'label':'Z'})])
        self.add_edges_from([(1, 5, {'weight':self.jz, 'label':'Z'})])
        self.add_edges_from([(3, 7, {'weight':self.jz, 'label':'Z'})])

        self.add_edges_from([(0, 1, {'weight':self.jx, 'label':'X'})])
        self.add_edges_from([(2, 3, {'weight':self.jx, 'label':'X'})])
        self.add_edges_from([(0, 3, {'weight':self.jy, 'label':'Y'})])
        self.add_edges_from([(2, 1, {'weight':self.jy, 'label':'Y'})])

        for e in self.edges: 
            term = ['I' for _ in range(self.number_of_spins)]
            term[e[0]],    term[e[1]] = self.edges[e]['label'], self.edges[e]['label']
            mag = self.J[self.edge_dict[self.edges[e]['label']] - 1]
            self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term[::-1]), mag=-1*mag)
                
        return None 

    def square_octagon_torus(self): 
        self.edge_direction = self.edge_direction_square_octagon
        self.site_qubit_label = self.site_qubit_label_square_octagon
        self.edge_qubit_label = self.edge_qubit_label_square_octagon
        self.number_of_spins = self.number_of_unit_cells*4
        self.spin_hamiltonian = {}
        for j in range(self.Ly): 
            for i in range(self.Lx):
                # create edges of the graph
                cell_indx = self.unit_cell_indx(i, j)
                node_0, node_1 = 4*cell_indx, + 4*cell_indx + 1
                node_2, node_3 = 4*cell_indx + 2, + 4*cell_indx + 3
                self.add_edges_from([(node_3, node_0, {'weight':self.jx, 'label':'X'}),
                                     (node_1, node_2, {'weight':self.jx, 'label':'X'}), 
                                     (node_0, node_1, {'weight':self.jy, 'label':'Y'}), 
                                     (node_2, node_3, {'weight':self.jy, 'label':'Y'})])
                
                right_cell_indx = self.unit_cell_indx(i + 1, j)
                right_node = 4*right_cell_indx + 1 
                self.add_edges_from([(right_node, node_3, {'weight':self.jz, 'label':'Z'})])

                above_cell_indx = self.unit_cell_indx(i, j+1)
                above_node = 4*above_cell_indx 
                self.add_edges_from([(node_2, above_node, {'weight':self.jz, 'label':'Z'})])

                term_1 = ['I' for _ in range(self.number_of_spins)]
                term_2 = ['I' for _ in range(self.number_of_spins)]
                term_3 = ['I' for _ in range(self.number_of_spins)]
                term_4 = ['I' for _ in range(self.number_of_spins)]
                term_5 = ['I' for _ in range(self.number_of_spins)]
                term_6 = ['I' for _ in range(self.number_of_spins)]
                
                term_1[node_3], term_1[node_0] = 'X', 'X'
                term_2[node_1], term_2[node_2] = 'X', 'X'
                term_3[node_0], term_3[node_1] = 'Y', 'Y'
                term_4[node_2], term_4[node_3] = 'Y', 'Y'
                term_5[right_node], term_5[node_3] = 'Z', 'Z'
                term_6[above_node], term_6[node_2] = 'Z', 'Z'

                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_1[::-1]), mag=-1*self.jx)
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_2[::-1]), mag=-1*self.jx)
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_3[::-1]), mag=-1*self.jy)
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_4[::-1]), mag=-1*self.jy)
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_5[::-1]), mag=-1*self.jz)
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_6[::-1]), mag=-1*self.jz)

        return None


    def square_octagon_open(self): 
        self.edge_direction = self.edge_direction_square_octagon
        self.site_qubit_label = self.site_qubit_label_square_octagon
        self.edge_qubit_label = self.edge_qubit_label_square_octagon
        self.number_of_spins = self.number_of_unit_cells*4
        self.spin_hamiltonian = {}
        for j in range(self.Ly): 
            for i in range(self.Lx):
                # create edges of the graph
                cell_indx = self.unit_cell_indx(i, j)
                node_0, node_1 = 4*cell_indx, + 4*cell_indx + 1
                node_2, node_3 = 4*cell_indx + 2, + 4*cell_indx + 3
                self.add_edges_from([(node_3, node_0, {'weight':self.jx, 'label':'X'}),
                                     (node_1, node_2, {'weight':self.jx, 'label':'X'}), 
                                     (node_0, node_1, {'weight':self.jy, 'label':'Y'}), 
                                     (node_2, node_3, {'weight':self.jy, 'label':'Y'})])
                
                right_cell_indx = self.unit_cell_indx(i + 1, j)
                right_node = 4*right_cell_indx + 1
                if (i+1) % self.Lx !=0:
                    self.add_edges_from([(right_node, node_3, {'weight':self.jz, 'label':'Z'})])
                    term_5 = ['I' for _ in range(self.number_of_spins)]
                    term_5[right_node], term_5[node_3] = 'Z', 'Z'
                    self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_5[::-1]), mag=-1*self.jz)
                else: 
                    self.add_edges_from([(right_node, node_3, {'weight':0, 'label':'Z'})])

                above_cell_indx = self.unit_cell_indx(i, j+1)
                above_node = 4*above_cell_indx 
                if (j+1) % self.Ly !=0:
                    term_6 = ['I' for _ in range(self.number_of_spins)]
                    term_6[above_node], term_6[node_2] = 'Z', 'Z'
                    self.add_edges_from([(node_2, above_node, {'weight':self.jz, 'label':'Z'})])
                    self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_6[::-1]), mag=-1*self.jz)
                else:
                    self.add_edges_from([(node_2, above_node, {'weight':0, 'label':'Z'})])


                term_1 = ['I' for _ in range(self.number_of_spins)]
                term_2 = ['I' for _ in range(self.number_of_spins)]
                term_3 = ['I' for _ in range(self.number_of_spins)]
                term_4 = ['I' for _ in range(self.number_of_spins)]
                
                term_1[node_3], term_1[node_0] = 'X', 'X'
                term_2[node_1], term_2[node_2] = 'X', 'X'
                term_3[node_0], term_3[node_1] = 'Y', 'Y'
                term_4[node_2], term_4[node_3] = 'Y', 'Y'

                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_1[::-1]), mag=-1*self.jx)
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_2[::-1]), mag=-1*self.jx)
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_3[::-1]), mag=-1*self.jy)
                self.spin_hamiltonian = self.add_term_to_hamiltonian(h=self.spin_hamiltonian, 
                                            term=''.join(term_4[::-1]), mag=-1*self.jy)
                
        return None

    def std_gauge(self):
        """Get the standard gauge. u[i,j] = +1 for i even and j odd and connected by an edge, and zero otherwise. 
        The matrix u is antisymmetric, u[i,j] = -u[j,i].
        Args:
            KM (KitaevModel): An instance of the KitaevModel class 

        Returns:
            ndarray: array of the standard gauge
        """
        u = zeros((self.number_of_spins, self.number_of_spins))
        for e in self.edges: 
            i, j = self.edge_direction(e)
            u[i, j] = self.edges[e]['weight']
            u[j, i] = -self.edges[e]['weight']

        return u 

    def jw_hamiltonian_u(self, u):
        """Get the Hamiltonian of the system with a fixed gauge transformed using Jordan-Wigner 

        Args:
            u (ndarray): A matrix holding the gauge information such that u[i,j] is the gauge value in the (i,j) link. 

        Returns:
            dict: A dictionary of the Hamiltonian, eg {'IIIZ':0.5,'IIZI:-1} means h = 0.5 * IIIZ - IIZI.
        """

        h = {}
        for e in self.edges: 
            # the direction of the edge is important to know whether we have 
            # c_i c_j for c_j c_i in the Hamiltonian, they are not the same. 
            i, j = self.edge_direction(e)
            ip, jp = self.site_qubit_label(i), self.site_qubit_label(j)
            mag = self.edges[e]['weight'] * u[i,j]
            term = ['I' for _ in range(self.number_of_Dfermions_u)]
            if ip == jp: 
                term[ip] = 'Z'
                mag = -1*mag
            # The way I define things is such that for the 'c_i' fermions, even i  
            # will correspons to a X operator where as an odd j corresponds to a Y operator. 
            # However it does matter whether or not ip > jp. (the label of the qubit). 
            # This is because of the tail of Z's in the JW transformation, which can turn an X 
            # to Y or Y to an X if the tail of Z's of the other operator hit them. 
            elif ip>jp:
                # for the 4-8-8 model it's not always true that i is even and j is odd. 
                # It could be that both are even or both are odd. 
                if i % 2 == 0:  
                    term[ip] = 'X'
                else: 
                    term[ip] = 'Y'
                if j % 2 == 0: 
                    term[jp] = 'Y'
                    mag = -1*mag
                else: 
                    term[jp] = 'X'
            else: 
                mag = -1*mag 
                if j % 2 == 0:  
                    term[jp] = 'X'
                else: 
                    term[jp] = 'Y'
                if i % 2 == 0: 
                    term[ip] = 'Y'
                    mag = -1*mag
                else: 
                    term[ip] = 'X'

            for k in range(min(ip,jp)+1, max(ip,jp)):
                term[k] = 'Z'
            h = self.add_term_to_hamiltonian(h=h, term=''.join(term[::-1]), mag=mag)
            
        return h


    def jw_hamiltonian(self):
        h = {}
        for e in self.edges: 
            # the direction of the edge is important to know whether we have 
            # u_{ij} c_i c_j for u_{ij} c_j c_i in the Hamiltonian, they are not the same. 
            # also note here that u_{ij} is an operator 
            i, j = self.edge_direction(e)
            ip, jp = self.site_qubit_label(i), self.site_qubit_label(j)
            edge_indx = self.edge_qubit_label(e)
            mag = -1*self.edges[e]['weight'] # need to check this sign though
            term = ['I' for _ in range(self.number_of_Dfermions)]
            if ip == jp: 
                term[ip] = 'Z'
                mag = -1*mag
            # The way I define things is such that for the 'c_i' fermions, even i  
            # will correspons to a X operator where as an odd j corresponds to a Y operator. 
            # However it does matter whether or not ip > jp. (the label of the qubit). 
            # This is because of the tail of Z's in the JW transformation, which can turn an X 
            # to Y or Y to an X if the tail of Z's of the other operator hit them. 
            elif ip>jp:
                # for the 4-8-8 model it's not always true that i is even and j is odd. 
                # It could be that both are even or both are odd. 
                if i % 2 == 0:  
                    term[ip] = 'X'
                else: 
                    term[ip] = 'Y'
                if j % 2 == 0: 
                    term[jp] = 'Y'
                    mag = -1*mag
                else: 
                    term[jp] = 'X'
            else: 
                mag = -1*mag 
                if j % 2 == 0:  
                    term[jp] = 'X'
                else: 
                    term[jp] = 'Y'
                if i % 2 == 0: 
                    term[ip] = 'Y'
                    mag = -1*mag
                else: 
                    term[ip] = 'X'

            for k in range(min(ip,jp)+1, max(ip,jp)):
                term[k] = 'Z'
            
            # add the u_{ij} part
            term[edge_indx] = 'Z'
            h = self.add_term_to_hamiltonian(h=h, term=''.join(term[::-1]), mag=mag)

            # add magnetic field terms if they exist  
            if self.magnetic_field != (0,0,0): 
                term_1 = ['I' for _ in range(self.number_of_Dfermions)]
                term_2 = ['I' for _ in range(self.number_of_Dfermions)]
                mag = self.magnetic_field[self.edge_dict[self.edges[e]['label']] - 1] # need to check sign
                term_1[edge_indx] = 'X'
                if i % 2 == 0 :
                    term_1[ip] = 'Y'
                    mag = -mag
                else: 
                    term_1[ip] = 'X'
                    # mag = -mag

                for k in range(min(ip,edge_indx)+1, max(ip,edge_indx)):
                    term_1[k] = 'Z'
                h = self.add_term_to_hamiltonian(h=h, term=''.join(term_1[::-1]), mag=mag)

                mag = -1*self.magnetic_field[self.edge_dict[self.edges[e]['label']] - 1]
                term_2[edge_indx] = 'Y'
                if j % 2 == 0: 
                    term_2[jp] = 'Y'
                    # mag = -mag
                else: 
                    term_2[jp] = 'X'
                    mag = -mag

                for k in range(min(jp,edge_indx)+1, max(jp,edge_indx)):
                    term_2[k] = 'Z'
                h = self.add_term_to_hamiltonian(h=h, term=''.join(term_2[::-1]), mag=mag)
                
        return h