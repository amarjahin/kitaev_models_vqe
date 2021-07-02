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
from cost import phys_energy_ev
from reduce_ansatz import reduce_params, reduce_ansatz
from projector_op import projector

mes = NumPyMinimumEigensolver()


def init_state(m, init_gauge=[]):
    """Initialize the qubit in the gauge free configuration.

    Args:
        m (int): Number of qubits in the circuit 
        init_gauge (list, optional): A list of the gauge qubits to be flipped. Defaults to [].

    Returns:
        QuantumCircuit: A quantum circuit that does the initialization
    """
    qc = QuantumCircuit(m, name='init_state') 
    if len(init_gauge) != 0:
        qc.x(init_gauge)
    return qc


def change_ansatz(m, init_gauge_p): 
    qc = QuantumCircuit(m, name='init_state') 
    if len(init_gauge_p) != 0:
        qc.x(init_gauge_p)

    for q in init_gauge_p: 
        for i in range(0, q): 
            qc.z(i)
    return qc

L = (1,1)           # size of the lattice 
J = (1.0, 1.0, 1.0) # pure Kitaev terms 
H = (0.1, 0.1, 0.1)   # magnetic terms 

# choose the kind of lattice and boundary conditions
# lattice_type = 'honeycomb_open'
# lattice_type = 'honeycomb_torus'
# lattice_type = 'eight_spins_4_8_8'
lattice_type = 'square_octagon_torus'
# lattice_type = 'square_octagon_open'

FH = KitaevModel(L=L, J=J, H=H, lattice_type=lattice_type) # this class contain various information about the model

m_u = FH.number_of_Dfermions_u
m = FH.number_of_Dfermions
active_qubits = [*range(m)]
fermions_qubits = [*range(m_u)]
gauge_qubits = [*range(m_u, m)]
# init_gauge = [*range(m_u, m)]
# init_gauge_p = []
###### use this for when L = (1,1)##########
#### this is to get ground state ######
# init_gauge = [2,3,4,5, 7]
# init_gauge_p = [6]
#### this is to get first excited state ######
init_gauge = [2,3,4,5, 6]
init_gauge_p = [7]
############################################

###### use this for when L = (2,1)##########
# init_gauge = [*range(m_u, m)]
# init_gauge_p = [15]
# for i in init_gauge_p: 
#     init_gauge.remove(i)
############################################

h = FH.jw_hamiltonian() # the Jordan_Wigner transformed fermionic Hamiltonian
qubit_op = convert_to_qiskit_PauliSumOp(h)
# hamiltonian = qubit_op.to_matrix(massive = True)
# hamiltonian = qubit_op.to_spmatrix()
# exact_eigenvalues, exact_eigenstates = eigh(hamiltonian)
# exact_eigenvalues, exact_eigenstates = eigsh(hamiltonian, k=1, which='SA')
fermion_result = mes.compute_minimum_eigenvalue(qubit_op)
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

qc = QuantumCircuit(m)
qc.append(init_state(m, init_gauge=init_gauge).to_instruction(), qargs=active_qubits)
qc = transpile(qc, simulator)
qc.barrier()
last_element = qc.data[-1] # use this to keep track of where various parts of the circuit start and end

projector_op = projector(FH)
# projector_mat = projector_op.to_matrix()
projector_mat = projector_op.to_spmatrix()

# init_sim = simulator.run(qc).result()
# init_state_vec = init_sim.get_statevector()
# init_phys_component = conjugate(init_state_vec.T) @ projector_mat @ init_state_vec 

init_phys_component = phys_energy_ev(hamiltonian=qubit_op.to_spmatrix() ,simulator=simulator,
                qc_c=qc,params=[], projector= projector_op.to_spmatrix(), return_projection=True)[1].real
print(f'initial phys projection: {init_phys_component}')

# cost = lambda params: phys_energy_ev(hamiltonian=qubit_op ,simulator=simulator,
#                 qc_c=qc,params=params, projector= projector_op).real

cost = lambda params: phys_energy_ev(hamiltonian=qubit_op.to_spmatrix() ,simulator=simulator,
                qc_c=qc,params=params, projector= projector_op.to_spmatrix()).real

print(f'the initial energy: {cost([])}')

if round(init_phys_component, 8) == 0: 
    raise ValueError('no phys component')
det = 1
# A dictionary holding the circuit used for various set of terms of the ansatz
# 'a' --> theta_{ij} c_i c_j 
# 'b' --> theta^alpha_{ij} b^alpha_i c_j 
# 'c' --> theta^{alpha beta}_{ij} b^alpha_i b^beta_j 
# 'd' --> theta^{alpha beta}_{ijkl} b^alpha_i b^beta_j c_k c_l 
ansatz_terms_dict = {'a': GBSU(num_qubits=m, active_qubits=fermions_qubits, det=1, steps=1,param_name='a'), 
        'b': PSU(num_qubits=m, gauge_qubits=gauge_qubits, fermion_qubits=fermions_qubits,det=det, param_name='b'), 
        'c': GBSU(num_qubits=m, active_qubits=gauge_qubits, det=det, steps=1, param_name='c'), 
        'd': PDU(num_qubits=m, gauge_qubits=gauge_qubits, fermion_qubits=fermions_qubits, param_name='d')
        }
num_terms_grouped = {'a':2, 'b':2, 'c':2, 'd':8}
params0 = []
red_op_params = []
num_old_params = 0
nfev = 0
nit = 0
print(r"Multistep VQE start")
print(r"'a' terms are $\theta_{ij} c_i c_j$")
print(r"'b' terms are $\theta^{\alpha}_{ij} b^\alpha_i c_j$")
print(r"'c' terms are $\theta^{\alpha \beta}_{ij} b^\alpha_i b^\beta_j$")
print(r"'d' terms are $\theta^{\alpha \beta}_{ijkl} b^\alpha_i b^\beta_j  c_k c_l$")
for key in ansatz_terms_dict:
    # if key =='d': 
    #     break 

    qc.append(change_ansatz(m, init_gauge_p=init_gauge_p).to_instruction(), qargs=active_qubits)
    qc.append(ansatz_terms_dict[key].to_instruction(),qargs=active_qubits) # add next set of terms
    qc.append(change_ansatz(m, init_gauge_p=init_gauge_p).to_instruction(), qargs=active_qubits)
    
    print(f"num parameters after adding the '{key}' terms: {qc.num_parameters}")
    qc = transpile(qc, simulator) 
    params0 = list(zeros(qc.num_parameters))
    params0[0:len(red_op_params)] = red_op_params 
    print('optimizer is now running...')
    result = minimize(fun=cost, x0=params0,  method=method, tol=0.0001, options={'maxiter':12}) # run optimizer
    nfev = nfev + result['nfev']
    nit = nit + result['nit']
    print(f"optimization success:{result['success']}")
    op_params = list(result['x']) # get optimal params 
    qc = reduce_ansatz(qc, op_params, num_terms=num_terms_grouped[key], 
                        num_old_params=num_old_params, last_element=last_element)
    qc.barrier()
    last_element = qc.data[-1]
    red_op_params = reduce_params(op_params, num_old_params)
    num_old_params = qc.num_parameters

    if not result['success']: # in case the optimizer was interupted, run optimizer again
        print(f"num parameters after parameters reduction: {qc.num_parameters}")
        print('optimizer running again after reduction...')
        result = minimize(fun=cost, x0=red_op_params,  method=method, tol=0.0001, options={'maxiter':None})
        nfev = nfev + result['nfev']
        nit = nit + result['nit']
        print(f"optimization success:{result['success']}")
        op_params = list(result['x'])
        qc = reduce_ansatz(qc, op_params, num_terms=num_terms_grouped[key], 
                            num_old_params=num_old_params, last_element=last_element)
        qc.barrier()
        last_element = qc.data[-1]
        red_op_params = reduce_params(op_params, num_old_params)
        num_old_params = qc.num_parameters

    print(f"final num reduced parameters after adding the '{key}' terms: {qc.num_parameters}")
    print(f"The optimal energy after adding the '{key}' terms: {result['fun'] }")


optimal_energy = result['fun']

print(f'optimal energy: {optimal_energy}')
print(f"energy % error: {(optimal_energy - spin_result.eigenvalue.real) / spin_result.eigenvalue.real}")

print('num of iterations: ', nit)

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
