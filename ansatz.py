from qiskit import QuantumCircuit 
from qiskit.circuit import Parameter, ParameterVector
from numpy import pi

def GSU(num_qubits, active_qubits,det=1, steps=1):
    m = len(active_qubits)
    # defining the parameters used in this ansatz. 
    theta_1 = ParameterVector('th_1', length=int(m*(m+1)/2))
    theta_2 = ParameterVector('th_2', length=int(m*(m-1)/2))
    theta_3 = ParameterVector('th_3', length=int(m*(m-1)/2))
    theta_4 = ParameterVector('th_4', length=int(m*(m-1)/2))

    qc = QuantumCircuit(num_qubits)
    for _ in range(steps):
        index_1 = 0
        index_2 = 0
        index_3 = 0
        index_4 = 0
        active_qubits_p = active_qubits.copy()
        for i in active_qubits:
            active_qubits_p.pop(0)
            # a^dagger_i a_i 
            qc.rz((1/steps) * theta_1[index_1] , i)   
            index_1 = index_1 + 1  
            for j in active_qubits_p:
                # a^dagger_i a_j + a^dagger_j a_i 
                qc.h([i, j]) 
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz( (1/steps)*theta_1[index_1]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([i, j]) 
 
                qc.rx(pi/2, [i, j])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz( (1/steps)*theta_1[index_1]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.rx(-pi/2, [i, j])

                index_1 = index_1 + 1  

                #################################################
                # a^dagger_i a_j - a^dagger_j a_i 
                qc.h([i]) 
                qc.rx(pi/2, [j])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz((1/steps)*theta_2[index_2]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz(-(1/steps)*theta_2[index_2]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([j]) 
                qc.rx(-pi/2, [i])

                index_2 = index_2 + 1  

                #####################################################
                # a_i a_j + a^dagger_j a^dagger_i 
                qc.h([i, j]) 
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz( (1/steps)*theta_3[index_3]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([i, j]) 
 
                qc.rx(pi/2, [i, j])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz(- (1/steps)*theta_3[index_3]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.rx(-pi/2, [i, j])

                index_3 = index_3 + 1  

                ######################################################
                # a_i a_j - a^dagger_j a^dagger_i 
                qc.h([i]) 
                qc.rx(pi/2, [j])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz((1/steps)*theta_4[index_4]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz((1/steps)*theta_4[index_4]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([j]) 
                qc.rx(-pi/2, [i])

                index_4 = index_4 + 1   
    if det == -1:
        qc.x([0])
    return qc
    
def GBSU(num_qubits, active_qubits, det=1, steps=1, param_name='th'):
    m = len(active_qubits)
    # defining the parameters used in this ansatz. 
    theta_3 = ParameterVector(f'{param_name}_3', length=int(m*(m-1)/2))
    theta_4 = ParameterVector(f'{param_name}_4', length=int(m*(m-1)/2))

    qc = QuantumCircuit(num_qubits)
    for _ in range(steps):
        index_3 = 0
        index_4 = 0
        active_qubits_p = active_qubits.copy()
        for i in active_qubits:
            active_qubits_p.pop(0)
            for j in active_qubits_p: 
                #####################################################
                # a_i a_j + a^dagger_j a^dagger_i 
                qc.h([i, j]) 
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz( (1/steps)*theta_3[index_3]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([i, j]) 
 
                qc.rx(pi/2, [i, j])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz(-(1/steps)*theta_3[index_3]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.rx(-pi/2, [i, j])

                index_3 = index_3 + 1  

                # a_i a_j - a^dagger_j a^dagger_i 
                qc.h([i]) 
                qc.rx(pi/2, [j])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz((1/steps)*theta_4[index_4]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz((1/steps)*theta_4[index_4]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([j]) 
                qc.rx(-pi/2, [i])

                index_4 = index_4 + 1  
    
    if det == -1:
        qc.x([0])
    return qc


##########################################
########## UNDER CONSTRUCTION ############
##########################################
def mix_gauges(num_qubits):
    phi = ParameterVector('phi', length=num_qubits)
    # ang = ParameterVector('ang', length=num_qubits)
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    for i in range(num_qubits):
        # qc.cx(i, i+1)
        qc.ry(phi[i], i)
        # qc.rx(ang[i], i)
        # qc.cx(i, i+1)
    qc.h(range(num_qubits))
    return qc

def PGBSU(num_qubits, fermions_qubits, det=1, steps=1, param_name='th'):
    # m = len(active_qubits)
    m = len(fermions_qubits)
    active_qubits = [*range(num_qubits)]
    # defining the parameters used in this ansatz. 
    length = 0 
    for i in range(m): 
        for _ in range(i+1, num_qubits): 
            length = length +1

    theta_3 = ParameterVector(f'{param_name}_3', length=length)
    theta_4 = ParameterVector(f'{param_name}_4', length=length)

    qc = QuantumCircuit(num_qubits)
    for _ in range(steps):
        index_3 = 0
        index_4 = 0
        # active_qubits_p = active_qubits.copy()
        for i in fermions_qubits:
            active_qubits.pop(0)
            for j in active_qubits: 
                #####################################################
                # a_i a_j + a^dagger_j a^dagger_i 
                qc.h([i, j]) 
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz( (1/steps)*theta_3[index_3]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([i, j]) 
 
                qc.rx(pi/2, [i, j])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz(-(1/steps)*theta_3[index_3]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.rx(-pi/2, [i, j])

                index_3 = index_3 + 1  

                # a_i a_j - a^dagger_j a^dagger_i 
                qc.h([i]) 
                qc.rx(pi/2, [j])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz((1/steps)*theta_4[index_4]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                for k in range(i, j): 
                    qc.cx(k, k+1)
                qc.rz((1/steps)*theta_4[index_4]/2, j)
                for k in range(i, j)[::-1]: 
                    qc.cx(k, k+1)
                qc.h([j]) 
                qc.rx(-pi/2, [i])

                index_4 = index_4 + 1  
    
    if det == -1:
        qc.x([0])
    return qc
