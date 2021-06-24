import time 
from numpy import pi
from numpy.random import random, rand
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import StatevectorSimulator, UnitarySimulator, QasmSimulator

start = time.time()
num_params = 86
# number of terms != number of paramaters because I group some of the terms
# under the same parameter 
num_terms= 2*56 + 8*30 
params = ParameterVector('a', length = num_params)
qc = QuantumCircuit(8)
# This circuit is not the ansatz, but it looks like it. 
# this construction of the circuit is worst that what it actually is in 
# terms of the number of cx gates. I expect the actual 
for i in range(num_terms):
    for j in range(7): 
        qc.cx(j, j+1)
    if i < 86:
        qc.rz(params[i], 7)
    else:
        qc.rz(pi/4, 7)
    for j in range(7)[::-1]: 
        qc.cx(j, j+1)
    qc.h([*range(8)])

qc = transpile(qc)  
end = time.time()
print(f'time took to build the circuit: {end - start}')
nfev = 4279 # this is the number of times the optimizer calls the cost function. 
# Just make the calling to the cost function as many times as the optimizer.
for i in range(nfev): 
    params = pi*random(num_params)
    qc_c = qc.bind_parameters(params)
    simulator = StatevectorSimulator()
    result = simulator.run(qc_c).result()

    if i % 100 == 0: 
        print(f'time took to call the function {i+1} times: {time.time() - end}')