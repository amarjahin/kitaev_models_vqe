from numpy import conjugate
from qiskit import QuantumCircuit, transpile

def energy_ev(hamiltonian,simulator, qc_c, params, projector):
    
    qc_c = qc_c.bind_parameters(params)
    result = simulator.run(qc_c).result()
    state_vec = result.get_statevector()
    phys_prop = conjugate(state_vec.T) @ projector @ state_vec
    # energy_ev = (conjugate(state_vec.T) @ hamiltonian @ state_vec).real 
    phys_energy_ev = (conjugate(state_vec.T) @ projector @ hamiltonian @ state_vec).real/(phys_prop)
    return phys_energy_ev