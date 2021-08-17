from numpy import zeros, array, conjugate, pi, round, savetxt, loadtxt, linspace, zeros_like
from numpy.linalg import norm, eigh
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit.algorithms import NumPyEigensolver,NumPyMinimumEigensolver
from qiskit.circuit.library.standard_gates.rz import RZGate

from kitaev_models import KitaevModel
from qiskit_conversion import convert_to_qiskit_PauliSumOp
from ansatz import GBSU, PSU, PDU, PFDU, mix_gauge
from cost import phys_energy_ev, energy_ev

from reduce_ansatz import reduce_params, reduce_ansatz
# from projector_op import projector

mes = NumPyMinimumEigensolver()
# es = NumPyEigensolver(k=2)


def init_state(m, init_gauge=[]):
    """Initialize the qubit in the gauge free configuration.

    Args:
        m (int): Number of qubits in the circuit 
        init_gauge (list, optional): A list of the gauge qubits to be flipped. Defaults to [].

    Returns:
        QuantumCircuit: A quantum circuit that does the initialization
    """
    qc = QuantumCircuit(m, name='init_state') 
    if len(init_gauge) != 0:
        qc.x(init_gauge)
    return qc


def change_ansatz(m, init_gauge_p): 
    qc = QuantumCircuit(m, name='change_ansatz') 
    if len(init_gauge_p) != 0:
        qc.x(init_gauge_p)

    for q in init_gauge_p: # this is a jordan-wigner string
        for i in range(0, q): 
            qc.z(i)
    return qc

L = (2,1)           # size of the lattice 
J = (1.0, 1.0, 2**(0.5)) # pure Kitaev terms

# h_array = linspace(0, 0.2, num=21)
h_array = [0.1]
exact_energy_array = zeros_like(h_array)
optimal_energy_array = zeros_like(h_array)
state_overlap_array = zeros_like(h_array)
phys_prob_array = zeros_like(h_array)
nfev_array = zeros_like(h_array)
nit_array = zeros_like(h_array)
for l in range(len(h_array)):
    H = (h_array[l], h_array[l], h_array[l])   # magnetic terms 
    print('#################', h_array[l], '##################')
    # choose the kind of lattice and boundary conditions
    # lattice_type = 'honeycomb_torus'
    lattice_type = 'square_octagon_torus'

    FH = KitaevModel(L=L, J=J, H=H, lattice_type=lattice_type) # this class contain various information about the model

    m_u = FH.number_of_Dfermions_u
    m = FH.number_of_Dfermions
    active_qubits = [*range(m)]
    fermions_qubits = [*range(m_u)]
    gauge_qubits = [*range(m_u, m)]
    init_gauge = [*range(m_u, m)]

    edges=[(1,7)]
    # edges=[(2,0)]

    init_gauge_p = [FH.edge_qubit_label(e) for e in edges]
    ###### andy's ##########
    # init_gauge_p = [11, 4, 10, 5]
    # init_gauge_p = [10, 13]
    ############################################
    for i in init_gauge_p: 
        init_gauge.remove(i)
        
    h = FH.jw_hamiltonian() # the Jordan_Wigner transformed fermionic Hamiltonian
    qubit_op = convert_to_qiskit_PauliSumOp(h)
    hamiltonian = qubit_op.to_spmatrix()
    fermion_result = mes.compute_minimum_eigenvalue(qubit_op)
    print(f'exact fermion energy: {fermion_result.eigenvalue.real}')

    spin_ham = convert_to_qiskit_PauliSumOp(FH.spin_hamiltonian)
    spin_result = mes.compute_minimum_eigenvalue(spin_ham)
    print(f'exact spin energy: {spin_result.eigenvalue.real}')

    #######################################################################
    simulator = StatevectorSimulator()
    method = 'BFGS'

    qc = QuantumCircuit(m)
    qc.append(init_state(m, init_gauge=init_gauge).to_instruction(), qargs=active_qubits)
    qc = transpile(qc, simulator)
    qc.barrier()
    last_element = qc.data[-1] # use this to keep track of where various parts of the circuit start and end

    projector_op = FH.projector()
    projector_mat = projector_op.to_spmatrix()
    init_sim = simulator.run(qc).result()
    init_state_vec = init_sim.get_statevector()
    init_phys_component = conjugate(init_state_vec.T) @ projector_mat @ init_state_vec 

    print(f'initial phys projection: {init_phys_component}')

    cost = lambda params: phys_energy_ev(hamiltonian=hamiltonian ,simulator=simulator,
                    qc_c=qc,params=params, projector=projector_mat).real

    print(f'the initial energy: {cost([])}')

    if round(init_phys_component, 8) == 0: 
        raise ValueError('no phys component')
    det = 1
    ansatz_terms_dict = {'a': GBSU(num_qubits=m, active_qubits=fermions_qubits, det=1, steps=1,param_name='a'), 
            'b': GBSU(num_qubits=m, active_qubits=gauge_qubits, det=det, steps=1, param_name='b'), 
            # 'c': PSU(num_qubits=m, fermion_qubits=fermions_qubits, gauge_qubits=gauge_qubits, param_name='c'),
            # 'e': PFDU(num_qubits=m, fermion_qubits=fermions_qubits, steps=1, param_name='e'),
            # 'b': mix_gauge(num_qubits=m, gauge_qubits=gauge_qubits, param_name='b'),
            # 'e': PDU(num_qubits=m, gauge_qubits=gauge_qubits, fermion_qubits=fermions_qubits, param_name='e')
            }
    num_terms_grouped = {'a':2, 'b':2, 'c':2, 'd':8, 'e':8}
    params0 = []
    op_params = []
    num_old_params = 0
    nfev = 0
    nit = 0

    for key in ansatz_terms_dict:
        qc.append(change_ansatz(m, init_gauge_p=init_gauge_p).to_instruction(), qargs=active_qubits)
        qc.append(ansatz_terms_dict[key].to_instruction(),qargs=active_qubits) # add next set of terms
        qc.append(change_ansatz(m, init_gauge_p=init_gauge_p).to_instruction(), qargs=active_qubits)
        print(f"num parameters after adding the '{key}' terms: {qc.num_parameters}")
        qc = transpile(qc, simulator) 
        params0 = list(zeros(qc.num_parameters))
        params0[0:len(op_params)] = op_params
        print('optimizer is now running...')
        result = minimize(fun=cost, x0=params0,  method=method, tol=0.0001, options={'maxiter':None}) # run optimizer
        nfev = nfev + result['nfev']
        nit = nit + result['nit']
        op_params = result['x']
        print(f"optimization success:{result['success']}")
        print(f"The optimal energy: {result['fun'] }")

    print('num of iterations: ', nit)
    print('num of evaluations: ', nfev)

    if isinstance(simulator, StatevectorSimulator):
        op_qc = qc.bind_parameters(result['x'])
        op_state = simulator.run(op_qc).result().get_statevector()
        phys_op_state = projector_mat @ op_state
        phys_prob = norm(phys_op_state)**2
        print('<optimal|P|optimal>:', phys_prob)
        phys_op_state = projector_mat @ op_state/ norm(phys_op_state)
        # savetxt('vqe_results/andy_lattice/h_075/state_cccc.txt', phys_op_state)

    optimal_energy = (conjugate(phys_op_state.T) @ hamiltonian @ phys_op_state).real

    print(f'optimal energy: {optimal_energy}')
    print(f"optimal - exact / exact: {(optimal_energy - spin_result.eigenvalue.real)/abs(spin_result.eigenvalue.real)}")

    if L == (1,1):
        overlap_subspace = 0
        vals, vecs = eigh(qubit_op.to_matrix())
        for ip in range(len(vals)): 
            if round(optimal_energy, 10) == round(vals[ip], 10):
                prob = abs(conjugate(phys_op_state.T) @ vecs[:,ip])**2 
                overlap_subspace = overlap_subspace + prob
                print(ip, vals[ip])
            
        print(f"1 - |<exact|optimal>|^2 : {1 - overlap_subspace}")
    exact_energy_array[l] = spin_result.eigenvalue.real
    optimal_energy_array[l] = optimal_energy
    state_overlap_array[l] = 1 - overlap_subspace
    phys_prob_array[l] = phys_prob
    nfev_array[l] = nfev
    nit_array[l] = nit
    