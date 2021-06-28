from numpy import pi, conjugate, argsort, array 
from qiskit_conversion import convert_to_qiskit_PauliSumOp


def projection_op_i(KM, i): 
    """This gives the projection operator at the i-th site: 1/2*(1 + D_i) 

    Args:
        KM (KitaevModel): An instance of the class KitaevModel
        i (int): the site on which to get the projection 

    Returns:
        dict: Dictionary with the operator 1/2*(1+D_i)
    """
    h = {}
    edges = [(i, j) for j in KM[i]] 

    edges_labels = [KM.edge_qubit_label(edges[i]) for i in range(len(edges))]
    # edges_labels = sorted(edges_labels)
    sorting_inds = argsort(edges_labels)
    edges_labels = [edges_labels[i] for i in sorting_inds]
    edges = [edges[i] for i in sorting_inds]
    edges_directed = [KM.edge_direction(edges[i]) for i in range(len(edges))]
    correct_order = [edges_directed[i] == edges[i] for i in range(len(edges))]
    
    ip = i//2
    j = edges_labels[0]
    k = edges_labels[1]
    l = edges_labels[2]
    term = ['I' for _ in range(KM.number_of_Dfermions)]
    mag = 0.5
    h = KM.add_term_to_hamiltonian(h=h, term=''.join(term[::-1]), mag=mag)
    
    alpha = [KM.edges[edges[i]]['label'] for i in range(3)]
    if alpha == ['X', 'Z', 'Y'] or alpha == ['Y', 'X', 'Z'] or alpha == ['Z', 'Y', 'X']: 
        mag = -mag

    if i%2 == 0: 
        term[ip] = 'Y'
        mag = -mag
    else: 
        term[ip] = 'X'
    for kp in range(ip+1, j): 
        term[kp] = 'Z'
    if correct_order[0]: 
        term[j] = 'X'
    else: 
        term[j] = 'Y'

    if correct_order[1]: 
        term[k] = 'Y'
    else: 
        term[k] = 'X'
        mag = -mag

    for kp in range(k+1, l): 
        term[kp] = 'Z'
    if correct_order[2]: 
        term[l] = 'X'
    else: 
        term[l] = 'Y'
    # print(''.join(term[::-1]))

    h = KM.add_term_to_hamiltonian(h=h, term=''.join(term[::-1]), mag=mag)
    
    return h


def projector(KM):
    """This gives the projector operator onto the physical subspace prod_i 1/2*(1 + D_i)

    Args:
        KM (KitaevModel): An instance of KitaevModel

    Returns:
        ndarray: The projection operator
    """
    eye = ['I' for _ in range(KM.number_of_Dfermions)]

    projection_op = convert_to_qiskit_PauliSumOp( {''.join(eye):1} )

    for i in range(KM.number_of_spins):
        projection_op = projection_op @ convert_to_qiskit_PauliSumOp(projection_op_i(KM = KM, i=i))

    return projection_op