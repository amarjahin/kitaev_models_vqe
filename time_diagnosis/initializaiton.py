import time 
from numpy import pi
from numpy.random import random
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import StatevectorSimulator
from qiskit.quantum_info import Statevector


qc = QuantumCircuit(16)
for i in range(280):
    rz_pos = (i//2) % 14 + 1
    for j in range(rz_pos): 
        qc.cx(j, j+1)
    qc.rz(pi*random(1)[0], rz_pos)
    for j in range(rz_pos)[::-1]: 
        qc.cx(j, j+1)
    qc.h([*range(rz_pos)])

qc = transpile(qc)  
simulator = StatevectorSimulator()
result = simulator.run(qc).result()
state_vec = Statevector(result.get_statevector())

qc_1 = QuantumCircuit(16)
# qc_1.initialize(state_vec, [*range(16)]) # pylint: disable=no-member
qc_1.set_statevector(state_vec) # pylint: disable=no-member

qc_2 = QuantumCircuit(16)

start = time.time()
for i in range(1000): 
    result = simulator.run(qc_1).result()
time_1 = time.time() - start
print(f'time took to run circuit w initialization {1000} times: {round(time_1,3)}s')

start = time.time()
for i in range(1000): 
    result = simulator.run(qc_2).result()
time_2 = time.time() - start
print(f'time took to run circuit w/o initialization {1000} times: {round(time_2,3)}s')

