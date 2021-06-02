import os
os.environ['MPMATH_NOSAGE'] = 'true'

from numpy import zeros, pi
# setup for qiskit, VQE and the classical optimizer for it
from qiskit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
# setup for defining the system, and the Hamiltonian
from kitaev_models import KitaevModel
from qiskit_conversion import convert_to_qiskit_PauliSumOp
from ansatz import GSU,GBSU, mix_gauges


def get_gauge(KM):
    """Get the standard gauge. u[i,j] = +1 for i even and j odd and connected by an edge, and zero otherwise. 
       The matrix u is antisymmetric, u[i,j] = -u[j,i].
    Args:
        KM (KitaevModel): An instance of the KitaevModel class 

    Returns:
        ndarray: array of the standard gauge
    """
    u = zeros((KM.number_of_spins, KM.number_of_spins))
    for e in KM.edges: 
        i, j = KM.edge_direction(e)
        u[i, j] = KM.edges[e]['weight']
        u[j, i] = -KM.edges[e]['weight']

    return u 

def init_state(m, init_gauge=[]):
    qc = QuantumCircuit(m) 
    if len(init_gauge) != 0:
        qc.x(init_gauge)
    return qc

L = (2,2)
J = (1.0, 1.0, 1.0)
H = (0, 0, 0)
# lattice_type = 'honeycomb_open'
# lattice_type = 'honeycomb_torus'
# lattice_type = 'eight_spins_4_8_8'
# lattice_type = 'square_octagon_torus'
lattice_type = 'square_octagon_open'


FH = KitaevModel(L=L, J=J, H=H, lattice_type=lattice_type)
spin_hamiltonian = convert_to_qiskit_PauliSumOp(FH.spin_hamiltonian)
mes = NumPyMinimumEigensolver()
spin_result = mes.compute_minimum_eigenvalue(spin_hamiltonian)

# objects in tilde L_u
u = get_gauge(KM=FH)
h_u = FH.jw_hamiltonian_u(u=u)
qubit_op_u = convert_to_qiskit_PauliSumOp(h_u)
result_u = mes.compute_minimum_eigenvalue(qubit_op_u)

# objects in tilde L 
h = FH.jw_hamiltonian()
qubit_op = convert_to_qiskit_PauliSumOp(h)
# result = mes.compute_minimum_eigenvalue(qubit_op)



print(f'number of spins: {FH.number_of_spins}')
print(f"{'energy per unit cell from spin Hamiltonian using numpy:':<60}{spin_result.eigenvalue.real/FH.number_of_unit_cells:>10f}")
print(f"{'energy per unit cell from qubit_op using numpy:':<60}{result_u.eigenvalue.real/FH.number_of_unit_cells:>10f}")

#######################################################################
m_u = FH.number_of_Dfermions_u
det = 1

h_qubit_op = qubit_op_u
m = m_u
init_gauge = []
optimizer = COBYLA(maxiter=3000, tol=0.0001) 
simulator = StatevectorSimulator()
QI = QuantumInstance(backend=simulator)

# h_qubit_op = qubit_op
# m = FH.number_of_Dfermions
# init_gauge = [*range(m_u, m)]
# optimizer = SPSA(maxiter=300)
# simulator = QasmSimulator()
# QI = QuantumInstance(backend=simulator, shots=2000)

active_qubits = [*range(m_u)]
# active_qubits = [*range(m)]

# ansatz = GSU(num_qubits=m, active_qubits=[*range(m_u)],det=1, steps=1)
ansatz = GBSU(num_qubits=m, active_qubits=active_qubits, det=det, steps=1)
# ansatz = ansatz.compose(mix_gauges(num_qubits=FH.number_of_edges()), qubits=[*range(m_u, m)], front=True)

# add the initial state
ansatz =  ansatz.compose(init_state(m, init_gauge=init_gauge), front=True)


algorithm = VQE(ansatz,
                optimizer=optimizer,
                quantum_instance=QI , initial_point=zeros(ansatz.num_parameters)
                )

print(f'number of parameters: {ansatz.num_parameters}')
print(f'optimizing using ansatz with det = {det}')

vqe_result = algorithm.compute_minimum_eigenvalue(h_qubit_op)
# print(algorithm.optimal_params)
energy = vqe_result.eigenvalue.real

print(f"{'energy per unit cell using VQE:':<60}{energy/FH.number_of_unit_cells:>10f}")
print(f'number of iterations done: {algorithm._eval_count}')