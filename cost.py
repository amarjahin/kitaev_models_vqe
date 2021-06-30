import os
os.environ['MPMATH_NOSAGE'] = 'true'

from numpy import conjugate
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.opflow import CircuitStateFn, PauliExpectation, CircuitSampler, StateFn
from qiskit.utils import QuantumInstance


def phys_energy_ev(hamiltonian,simulator, qc_c, params, projector, return_projection=False):
    qc_c = qc_c.bind_parameters(params)
    if isinstance(simulator, StatevectorSimulator):
        hamiltonian = hamiltonian.to_matrix()
        projector = projector.to_matrix()
        result = simulator.run(qc_c).result()
        state_vec = result.get_statevector()
        phys_prop = conjugate(state_vec.T) @ projector @ state_vec
        # energy_ev = (conjugate(state_vec.T) @ hamiltonian @ state_vec).real 
        phys_energy_ev = (conjugate(state_vec.T) @ projector @ hamiltonian @ state_vec).real/(phys_prop)
        if not return_projection:
            return phys_energy_ev
        else: 
            return phys_energy_ev, phys_prop

    elif isinstance(simulator, QasmSimulator): 
        psi = CircuitStateFn(qc_c)
        QI = QuantumInstance(simulator, shots=10000)
        ms_ph = StateFn(projector @ hamiltonian, is_measurement=True) @ psi
        ms_p = StateFn(projector, is_measurement=True) @ psi
        ph_ev = PauliExpectation().convert(ms_ph)
        p_ev = PauliExpectation().convert(ms_p)
        ph_samp = CircuitSampler(QI).convert(ph_ev).eval().real
        p_samp = CircuitSampler(QI).convert(p_ev).eval().real
        if not return_projection:
            return ph_samp/p_samp
        else: 
            return ph_samp/p_samp, p_samp

    else: 
        raise ValueError('simulator can only be QasmSimulator or StatevectorSimulator')
