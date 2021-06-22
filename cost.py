from numpy import conjugate
from qiskit import QuantumCircuit, transpile

def energy_ev(hamiltonian,simulator, qc_c, params, phys_check=False, projector=None):
    
    qc_c = qc_c.bind_parameters(params)
    result = simulator.run(qc_c).result()
    state_vec = result.get_statevector()
    energy_ev = (conjugate(state_vec.T) @ hamiltonian @ state_vec).real
    if not phys_check: 
        return energy_ev
    else: 
        # this else is still highly experimental and can be just ignored at this point 
        phys_prob = conjugate(state_vec.T) @ projector @ state_vec 
        if phys_prob < 1e-3: 
            return 1e3 
            # return energy_ev

        else: 
            return energy_ev

