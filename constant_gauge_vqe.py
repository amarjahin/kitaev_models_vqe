from numpy import zeros, pi, conjugate 
# setup for qiskit, VQE and the classical optimizer for it
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.algorithms.optimizers import COBYLA
from qiskit.algorithms import VQE, NumPyMinimumEigensolver, NumPyEigensolver
# setup for defining the system, and the Hamiltonian
from kitaev_models import KitaevModel
from qiskit_conversion import convert_to_qiskit_PauliSumOp
from ansatz import GSU,GBSU

def change_gauge(u, edges): 
    for e in edges: 
        u[e[0], e[1]] = - u[e[0], e[1]]
        u[e[1], e[0]] = - u[e[1], e[0]]
    return u

es = NumPyEigensolver(k=2)

L = (2,2)
J = (1.0, 1.0, 1.0)
# lattice_type = 'honeycomb_open'
lattice_type = 'honeycomb_torus'
# lattice_type = 'eight_spins_4_8_8'
# lattice_type = 'square_octagon_torus'
# lattice_type = 'square_octagon_open'

FH = KitaevModel(L=L, J=J, H=(0,0,0), lattice_type=lattice_type)
spin_hamiltonian = convert_to_qiskit_PauliSumOp(FH.spin_hamiltonian)
spin_result = es.compute_eigenvalues(spin_hamiltonian)

u = FH.std_gauge()
# a change of gauge can be useful on a tours to change the non-contractible loops
# edges = [(7, 1), (11, 13), (0,14), (10,4)]
edges = [(3,2), (0,7), (5,6)]
u = change_gauge(u=u, edges=edges)
h_u = FH.jw_hamiltonian_u(u=u)
qubit_op = convert_to_qiskit_PauliSumOp(h_u)
result_u = es.compute_eigenvalues(qubit_op)
exact_eigenstate = result_u._eigenstates[0].to_matrix(massive=True)

print(f'lattice type: {lattice_type}')
print(f'number of spins: {FH.number_of_spins}')
print(f"{'energy per unit cell from spin Hamiltonian using numpy:':<60}",
        f"{spin_result.eigenvalues[0].real/FH.number_of_unit_cells:>10f}")
print(f"{'energy per unit cell from qubit_op using numpy:':<60}",
        f"{result_u.eigenvalues[0].real/FH.number_of_unit_cells:>10f}")
print(f"{'first excited state energy per unit cell:':<60}",
        f"{result_u.eigenvalues[1].real/FH.number_of_unit_cells:>10f}")

#######################################################################
optimizer = COBYLA(maxiter=5000, tol=0.0001) 
simulator = StatevectorSimulator()
QI = QuantumInstance(backend=simulator)

m = FH.number_of_Dfermions_u
active_qubits = [*range(m)]

det = -1
ansatz = GBSU(num_qubits=m, active_qubits=active_qubits, det=det, steps=1)

algorithm = VQE(ansatz,
                optimizer=optimizer,
                quantum_instance=QI , initial_point=zeros(ansatz.num_parameters)
                )

print(f'number of parameters: {ansatz.num_parameters}')
print(f'optimizing using ansatz with det = {det}')

vqe_result = algorithm.compute_minimum_eigenvalue(qubit_op)
# print(algorithm.optimal_params)
energy = vqe_result.eigenvalue.real

print(f"{'energy per unit cell using VQE:':<60}",
        f"{energy/FH.number_of_unit_cells:>10f}")
print(f'number of iterations done: {algorithm._eval_count}')

optimal_state = algorithm.get_optimal_vector()
overlap = abs(conjugate(optimal_state.T) @ exact_eigenstate)
print(f'|<exact|optimal>| = {overlap}')