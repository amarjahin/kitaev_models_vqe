from numpy import conjugate
from qiskit import QuantumCircuit, transpile

def energy_ev(hamiltonian,simulator, qc_c, params):
    
    qc_c = qc_c.bind_parameters(params)
    result = simulator.run(qc_c).result()
    state_vec = result.get_statevector()
    energy_ev = (conjugate(state_vec.T) @ hamiltonian @ state_vec).real 
    return energy_ev