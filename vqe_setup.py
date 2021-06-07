import os
os.environ['MPMATH_NOSAGE'] = 'true'

from numpy import zeros, pi, conjugate 
# setup for qiskit, VQE and the classical optimizer for it
from qiskit import QuantumCircuit, transpile
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.algorithms import VQE, NumPyMinimumEigensolver, NumPyEigensolver
from qiskit.opflow import StateFn
# setup for defining the system, and the Hamiltonian
from kitaev_models import KitaevModel
from qiskit_conversion import convert_to_qiskit_PauliSumOp
from ansatz import GSU,GBSU, mix_gauges, PGBSU

def init_state(m, init_gauge=[]):
    qc = QuantumCircuit(m) 
    if len(init_gauge) != 0:
        qc.x(init_gauge)
    return qc

es = NumPyEigensolver(k=16)

L = (1,1)
J = (1.0, 1.0, 1.0)
H = (0.1, 0.1, 0)
# lattice_type = 'honeycomb_open'
# lattice_type = 'honeycomb_torus'
# lattice_type = 'eight_spins_4_8_8'
# lattice_type = 'square_octagon_torus'
lattice_type = 'square_octagon_open'

FH = KitaevModel(L=L, J=J, H=H, lattice_type=lattice_type)
spin_hamiltonian = convert_to_qiskit_PauliSumOp(FH.spin_hamiltonian)
spin_result = es.compute_eigenvalues(spin_hamiltonian)

h = FH.jw_hamiltonian()
qubit_op = convert_to_qiskit_PauliSumOp(h)
result = es.compute_eigenvalues(qubit_op)
exact_eigenstates = result._eigenstates.to_matrix(massive=True)

print(f'lattice type: {lattice_type}')
print(f'number of spins: {FH.number_of_spins}')
print(f"{'energy per unit cell from spin Hamiltonian using numpy:':<60}",
        f"{spin_result.eigenvalues[0].real/FH.number_of_unit_cells:>10f}")
# print(f"{'first excited state energy per unit cell:':<60}",
#       f"{spin_result.eigenvalues[1].real/FH.number_of_unit_cells:>10f}")

print(f"{'energy per unit cell from qubit_op using numpy:':<60}",
        f"{result.eigenvalues[0].real/FH.number_of_unit_cells:>10f}")
# print(f"{'first excited state energy per unit cell:':<60}",
#       f"{result.eigenvalues[1].real/FH.number_of_unit_cells:>10f}")


#######################################################################
optimizer = COBYLA(maxiter=5000, tol=0.0001) 
simulator = StatevectorSimulator()
QI = QuantumInstance(backend=simulator)

# optimizer = SPSA(maxiter=300)
# simulator = QasmSimulator()
# QI = QuantumInstance(backend=simulator, shots=2000)

m_u = FH.number_of_Dfermions_u
m = FH.number_of_Dfermions
init_gauge = [*range(m_u, m)]

# active_qubits = [*range(m_u)]
active_qubits = [*range(m)]
fermions_qubits = [*range(m_u)]

det = 1
# ansatz = GSU(num_qubits=m, active_qubits=active_qubits,det=det, steps=1)
# ansatz = GBSU(num_qubits=m, active_qubits=active_qubits, det=det, steps=1)
ansatz = PGBSU(num_qubits=m,fermions_qubits=fermions_qubits , det=det, steps=1)
ansatz = ansatz.compose(mix_gauges(num_qubits=FH.number_of_edges()), qubits=[*range(m_u, m)])
# ansatz = ansatz.compose(evolve_mag(num_qubits=m))
# add the initial state
ansatz =  ansatz.compose(init_state(m, init_gauge=init_gauge), front=True)

algorithm = VQE(ansatz,
                optimizer=optimizer,
                quantum_instance=QI , initial_point=zeros(ansatz.num_parameters)
                )

print(f'number of parameters: {ansatz.num_parameters}')
print(f'optimizing using ansatz with det = {det}')

vqe_result = algorithm.compute_minimum_eigenvalue(qubit_op)
# print(algorithm.optimal_params)
energy = vqe_result.eigenvalue.real

print(f"{'energy per unit cell using VQE:':<60}{energy/FH.number_of_unit_cells:>10f}")
print(f'number of iterations done: {algorithm._eval_count}')

# optimal_qc = algorithm.get_optimal_circuit()
# optimal_qc = optimal_qc.append(evolve_mag(num_qubits=m), qargs=[*range(m)])
# optimal_qc_c = transpile(optimal_qc, backend=simulator)
# optimal_vec = StateFn(simulator.run(optimal_qc_c).result().get_statevector())
# optimal_energy = (~optimal_vec @ h_qubit_op @ optimal_vec).eval().real /FH.number_of_unit_cells
# print(f"{'energy per unit cell after mag_ev:':<60}{optimal_energy:>10f}")

optimal_state = algorithm.get_optimal_vector()
# overlap = abs(conjugate(optimal_state.T) @ exact_eigenstates[0])
# print(f'|<exact|optimal>| = {overlap}')

overlap_subspace = 0
for i in range(len(exact_eigenstates)): 
    prob = abs(conjugate(optimal_state.T) @ exact_eigenstates[i])**2 
    overlap_subspace = overlap_subspace + prob

print(f"|<exact|optimal>| = {overlap_subspace}")