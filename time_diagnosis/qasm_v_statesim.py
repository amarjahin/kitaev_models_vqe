import time 
from numpy import conjugate, pi
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.utils import QuantumInstance
from qiskit.opflow import CircuitStateFn, PauliExpectation, CircuitSampler, StateFn
from qiskit.opflow import PauliOp, PauliSumOp
from qiskit.quantum_info import Pauli

qubit_op = PauliOp(Pauli('IIZZIZZZIIZZIZZZ')) + PauliOp(Pauli('IIXXXZZXIIZZIZZZ')) 
start = time.time()
qc = QuantumCircuit(16)
for i in range(4):
    for j in range(15): 
        qc.cx(j, j+1)
    qc.rz(pi/2, 7)
    for j in range(15)[::-1]: 
        qc.cx(j, j+1)
    qc.h([*range(16)])

qc = transpile(qc)  

print(f'time took to build the circuit: {time.time() - start}')
start = time.time()
simulator = StatevectorSimulator()
for _ in range(10):
    result = simulator.run(qc).result()
    result = simulator.run(qc).result()

    psi = result.get_statevector()
    # qubit_mat = qubit_op.to_spmatrix()
    qubit_mat = qubit_op.to_spmatrix()
    qubit_mat = qubit_op.to_spmatrix()

    qubit_op_ev = conjugate(psi.T) @ qubit_mat @ psi
    qubit_op_ev = conjugate(psi.T) @ qubit_mat @ psi

t1 = time.time() - start
print(f'time to calculate expectation value using StatevectorSimulator: {round(t1, 4)}s')

start = time.time()
simulator = QasmSimulator()
QI = QuantumInstance(simulator, shots=2000)
for _ in range(10):
    psi = CircuitStateFn(qc)
    ms = StateFn(qubit_op, is_measurement=True) @ psi
    pe = PauliExpectation().convert(ms)
    samp = CircuitSampler(QI).convert(pe).eval().real
t2 = time.time() - start
print(f'time to calculate expectation value using Qasmsimulator: {round(t2, 4)}s')

print(f'QasmSimulator is {t2/t1} times slower than StatevectorSimulator')

