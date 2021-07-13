from numpy import zeros, array, conjugate, cos, sin, pi, round
from numpy.linalg import eigh
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh 
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.algorithms import NumPyEigensolver,NumPyMinimumEigensolver
from qiskit.circuit.library.standard_gates.rz import RZGate

from kitaev_models import KitaevModel
from qiskit_conversion import convert_to_qiskit_PauliSumOp
from ansatz import GBSU, PSU, PDU
from cost import phys_energy_ev, energy_ev
from reduce_ansatz import reduce_params, reduce_ansatz
# from projector_op import projector

mes = NumPyMinimumEigensolver()

def change_gauge(u, edges): 
    for e in edges: 
        u[e[0], e[1]] = - u[e[0], e[1]]
        u[e[1], e[0]] = - u[e[1], e[0]]
    return u


def get_full_state(psi_u, m_u, m, edges_label):
    psi = zeros(2**m, dtype=complex)
    i = 2**m 
    for e in edges_label: 
        i = i - 2**e

    psi[i-2**m_u:i] = psi_u
    return psi

L = (2,1)           # size of the lattice 
J = (1/2**0.5, 1/2**0.5, 1) # pure Kitaev terms 
# J = (1.0, 1.0, 1.0) # pure Kitaev terms 
H = (0, 0, 0)       # magnetic terms 

# choose the kind of lattice and boundary conditions
# lattice_type = 'honeycomb_open'
# lattice_type = 'honeycomb_torus'
# lattice_type = 'eight_spins_4_8_8'
lattice_type = 'square_octagon_torus'
# lattice_type = 'square_octagon_open'

FH = KitaevModel(L=L, J=J, H=H, lattice_type=lattice_type) # this class contain various information about the model

m_u = FH.number_of_Dfermions_u
m = FH.number_of_Dfermions
active_qubits = [*range(m_u)]

u = FH.std_gauge()
flp_edges = [(0,2)]
# flp_edges = [(0,14),(4,10)] # for L = (2,2)
# flp_edges = [(5,6), (4,7)] # for andy's 
# flp_edges = []
flp_edges_label = [FH.edge_qubit_label(e) for e in flp_edges]

u = change_gauge(u=u, edges=flp_edges)

h_u = FH.jw_hamiltonian_u(u) # the Jordan_Wigner transformed fermionic Hamiltonian
h = FH.jw_hamiltonian()      # the Jordan_Wigner transformed fermionic Hamiltonian

qubit_op_u = convert_to_qiskit_PauliSumOp(h_u)
qubit_op = convert_to_qiskit_PauliSumOp(h)
hamiltonian_u = qubit_op_u.to_spmatrix()
fermion_result = mes.compute_minimum_eigenvalue(qubit_op_u)
print(f'exact fermion energy: {fermion_result.eigenvalue.real}')

spin_ham = convert_to_qiskit_PauliSumOp(FH.spin_hamiltonian)
spin_result = mes.compute_minimum_eigenvalue(spin_ham)
print(f'exact spin energy: {spin_result.eigenvalue.real}')

#######################################################################
simulator = StatevectorSimulator()
# simulator = QasmSimulator()

# method = 'CG'
method = 'BFGS'
# method = 'SLSQP'

full_state = lambda psi_u:  get_full_state(psi_u, m_u, m, edges_label=flp_edges_label)

qc = QuantumCircuit(m_u)
# qc.barrier()
# last_element = qc.data[-1] # use this to keep track of where various parts of the circuit start and end

projector_op = FH.projector()
# projector_mat = projector_op.to_matrix()
projector_mat = projector_op.to_spmatrix()
result = simulator.run(qc).result()
init_vec_u = result.get_statevector()
init_vec = full_state(init_vec_u)
init_phys_component = conjugate(init_vec.T) @ projector_mat @ init_vec
print(f'initial phys projection: {init_phys_component}')

# cost = lambda params: phys_energy_ev(hamiltonian=hamiltonian ,simulator=simulator,
#                 qc_c=qc,params=params, projector= projector_mat, 
#                 full_state=full_state).real

cost = lambda params: energy_ev(hamiltonian=hamiltonian_u ,simulator=simulator,
                qc_c=qc,params=params).real
#########################################################################
print(f'the initial energy: {cost([])}')

# if round(init_phys_component, 8) == 0: 
#     raise ValueError('no phys component')
det = 1
# A dictionary holding the circuit used for various set of terms of the ansatz
# 'a' --> theta_{ij} c_i c_j 
ansatz_terms_dict = {'a': GBSU(num_qubits=m_u, active_qubits=active_qubits, det=1, steps=1,param_name='a')
        }
num_terms_grouped = {'a':2}
params0 = []
red_op_params = []
num_old_params = 0
nfev = 0
nit = 0
print(r"Multistep VQE start")
print(r"'a' terms are $\theta_{ij} c_i c_j$")
for key in ansatz_terms_dict:
    qc.append(ansatz_terms_dict[key].to_instruction(),qargs=active_qubits) # add next set of terms
    print(f"num parameters after adding the '{key}' terms: {qc.num_parameters}")
    qc = transpile(qc, simulator) 
    params0 = list(zeros(qc.num_parameters))
    params0[0:len(red_op_params)] = red_op_params 
    print('optimizer is now running...')
    result = minimize(fun=cost, x0=params0,  method=method, tol=0.0001, options={'maxiter':None}) # run optimizer
    nfev = nfev + result['nfev']
    nit = nit + result['nit']
    print(f"optimization success:{result['success']}")
    op_params = list(result['x']) # get optimal params 
    # qc = reduce_ansatz(qc, op_params, num_terms=num_terms_grouped[key], 
    #                     num_old_params=num_old_params, last_element=last_element)
    # qc.barrier()
    # last_element = qc.data[-1]
    # red_op_params = reduce_params(op_params, num_old_params)
    # num_old_params = qc.num_parameters

#     if not result['success']: # in case the optimizer was interupted, run optimizer again
#         print(f"num parameters after parameters reduction: {qc.num_parameters}")
#         print('optimizer running again after reduction...')
#         result = minimize(fun=cost, x0=red_op_params,  method=method, tol=0.0001, options={'maxiter':None})
#         nfev = nfev + result['nfev']
#         nit = nit + result['nit']
#         print(f"optimization success:{result['success']}")
#         op_params = list(result['x'])
#         qc = reduce_ansatz(qc, op_params, num_terms=num_terms_grouped[key], 
#                             num_old_params=num_old_params, last_element=last_element)
#         qc.barrier()
#         last_element = qc.data[-1]
#         red_op_params = reduce_params(op_params, num_old_params)
#         num_old_params = qc.num_parameters

    print(f"final num reduced parameters after adding the '{key}' terms: {qc.num_parameters}")
    print(f"The optimal energy after adding the '{key}' terms: {result['fun'] }")


optimal_energy = result['fun']

print(f'optimal energy: {optimal_energy}')
print(f"energy % error: {(optimal_energy - spin_result.eigenvalue.real) / spin_result.eigenvalue.real}")

print('num of iterations: ', nit)
print('num of evaluations: ', nfev)


# if isinstance(simulator, StatevectorSimulator):
#     op_qc = qc.bind_parameters(red_op_params)
#     op_state = simulator.run(op_qc).result().get_statevector()
#     op_state = (projector_mat @ op_state) / (conjugate(op_state.T) @ projector_mat @ op_state)**(0.5)

#     overlap_subspace = 0
#     for i in range(len(exact_eigenvalues)): 
#         if round(exact_eigenvalues[i].real, 10) == round(spin_result.eigenvalue.real, 10):
#             prob = abs(conjugate(op_state.T) @ exact_eigenstates[:,i])**2 
#             overlap_subspace = overlap_subspace + prob
#             print(i)
            
#     print(f"1 - |<exact|optimal>|^2 : {1 - overlap_subspace}")
