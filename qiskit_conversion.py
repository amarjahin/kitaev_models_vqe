from numpy import zeros
from qiskit.opflow import PauliOp, PauliSumOp
from qiskit.quantum_info import Pauli

def convert_to_qiskit_PauliSumOp(h={}):
    pauli_op = 0*PauliOp(Pauli(list(h)[0]))
    for key in h: 
        pauli_op = pauli_op + h[key] * PauliOp(Pauli(key))
    return pauli_op