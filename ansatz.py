from qiskit import QuantumCircuit 
from qiskit.circuit import Parameter, ParameterVector
from numpy import pi

def quadratic_exp(qc,theta, qubits): 
    i, j = qubits[0], qubits[1]
    for k in range(i, j): 
        qc.cx(k, k+1)
    qc.rz(theta, j)
    for k in range(i, j)[::-1]: 
        qc.cx(k, k+1)
    return None


def quartic_exp(qc, theta, qubits):
    i, j, ip, jp = qubits[0], qubits[1], qubits[2], qubits[3] 
    for k in range(i, j): 
        qc.cx(k, k+1)
    qc.cx(j, ip)
    for k in range(ip, jp): 
        qc.cx(k, k+1)
    qc.rz(theta, jp)
    for k in range(ip, jp)[::-1]: 
        qc.cx(k, k+1)
    qc.cx(j, ip)
    for k in range(i, j)[::-1]: 
        qc.cx(k, k+1)   
    return None

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
                ############################################
                # a^dagger_i a_j + a^dagger_j a_i 
                qc.h([i, j]) 
                quadratic_exp(qc=qc, theta=(1/steps)*theta_1[index_1]/2, qubits=[i,j])
                qc.h([i, j]) 
 
                qc.rx(pi/2, [i, j])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_1[index_1]/2, qubits=[i,j])
                qc.rx(-pi/2, [i, j])

                index_1 = index_1 + 1  

                #################################################
                # a^dagger_i a_j - a^dagger_j a_i 
                qc.h([i]) 
                qc.rx(pi/2, [j])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_2[index_2]/2, qubits=[i,j])
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                quadratic_exp(qc=qc, theta=-(1/steps)*theta_2[index_2]/2, qubits=[i,j])
                qc.h([j]) 
                qc.rx(-pi/2, [i])

                index_2 = index_2 + 1  

                #####################################################
                # a_i a_j + a^dagger_j a^dagger_i 
                qc.h([i, j]) 
                quadratic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.h([i, j]) 
 
                qc.rx(pi/2, [i, j])
                quadratic_exp(qc=qc, theta=-(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.rx(-pi/2, [i, j])

                index_3 = index_3 + 1  

                ######################################################
                # a_i a_j - a^dagger_j a^dagger_i 
                qc.h([i]) 
                qc.rx(pi/2, [j])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/2, qubits=[i,j])
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/2, qubits=[i,j])
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
                # a_j a_i + h.c
                qc.h([i, j]) 
                quadratic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.h([i, j]) 

                qc.rx(pi/2, [i, j])
                quadratic_exp(qc=qc, theta=-(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.rx(-pi/2, [i, j])

                index_3 = index_3 + 1  
                #####################################################
                # a_j a_i - h.c
                qc.h([i]) 
                qc.rx(pi/2, [j])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/2, qubits=[i,j])
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/2, qubits=[i,j])
                qc.h([j]) 
                qc.rx(-pi/2, [i])

                index_4 = index_4 + 1  
    
    if det == -1:
        qc.x([0])
    return qc

def PSU(num_qubits, gauge_qubits,fermion_qubits, det=1, steps=1, param_name='th', include_all=False):
    m_j, m_i = len(gauge_qubits), len(fermion_qubits)

    # defining the parameters used in this ansatz. 
    length = m_i * m_j
    theta_1 = ParameterVector(f'{param_name}_1', length=length)
    theta_2 = ParameterVector(f'{param_name}_2', length=length)
    theta_3 = ParameterVector(f'{param_name}_3', length=length)
    theta_4 = ParameterVector(f'{param_name}_4', length=length)

    qc = QuantumCircuit(num_qubits)
    for _ in range(steps):
        index_1 = 0
        index_2 = 0
        index_3 = 0
        index_4 = 0
        for i in fermion_qubits:
            for j in gauge_qubits: 
                if include_all==True:
                    ############################################
                    #  a_j a_i + h.c
                    qc.h([i, j]) 
                    quadratic_exp(qc=qc, theta=(1/steps)*theta_1[index_1]/2, qubits=[i,j])
                    qc.h([i, j]) 
    
                    qc.rx(pi/2, [i, j])
                    quadratic_exp(qc=qc, theta=-(1/steps)*theta_1[index_1]/2, qubits=[i,j])
                    qc.rx(-pi/2, [i, j])

                    index_1 = index_1 + 1  

                    #################################################
                    #  a_j a_i - h.c
                    qc.h([i]) 
                    qc.rx(pi/2, [j])
                    quadratic_exp(qc=qc, theta=(1/steps)*theta_2[index_2]/2, qubits=[i,j])
                    qc.h([i]) 
                    qc.rx(-pi/2, [j])

                    qc.h([j]) 
                    qc.rx(pi/2, [i])
                    quadratic_exp(qc=qc, theta=(1/steps)*theta_2[index_2]/2, qubits=[i,j])
                    qc.h([j]) 
                    qc.rx(-pi/2, [i])

                    index_2 = index_2 + 1  
                #####################################################
                # a^dagger_j a_i + h.c
                qc.h([i, j]) 
                quadratic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.h([i, j]) 

                qc.rx(pi/2, [i, j])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.rx(-pi/2, [i, j])

                index_3 = index_3 + 1  
                #####################################################
                # a^dagger_j a_i - h.c
                qc.h([i]) 
                qc.rx(pi/2, [j])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/2, qubits=[i,j])
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                quadratic_exp(qc=qc, theta=-(1/steps)*theta_4[index_4]/2, qubits=[i,j])
                qc.h([j]) 
                qc.rx(-pi/2, [i])

                index_4 = index_4 + 1  
    
    if det == -1:
        qc.x([0])
    return qc


def PDU(num_qubits, gauge_qubits,fermion_qubits, steps=1, param_name='th'): 
    m_j, m_i = len(gauge_qubits), len(fermion_qubits)
    # defining the parameters used in this ansatz. 
    length = (m_i * m_j)**2
    # theta_1 = ParameterVector(f'{param_name}_1', length=length)
    # theta_2 = ParameterVector(f'{param_name}_2', length=length)
    theta_3 = ParameterVector(f'{param_name}_3', length=length)
    theta_4 = ParameterVector(f'{param_name}_4', length=length)
    qc = QuantumCircuit(num_qubits)
    for _ in range(steps):
        # index_1 = 0
        # index_2 = 0
        index_3 = 0
        index_4 = 0
        fermion_qubits_p = fermion_qubits.copy()
        for i in fermion_qubits:
            fermion_qubits_p.pop
            for j in fermion_qubits:
                gauge_qubits_p = gauge_qubits.copy()
                for k in gauge_qubits:
                    gauge_qubits_p.pop(0)
                    for l in gauge_qubits_p:
                        #############################################
                        # a^dagger_l a^dagger_k a_j a_i + h.c
                        qc.h([i, j, k ,l]) 
                        quartic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/4, qubits=[i, j, k , l])
                        qc.h([i, j, k ,l]) 

                        qc.rx(pi/2, [i, j, k ,l]) 
                        quartic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/4, qubits=[i, j, k , l])
                        qc.rx(-pi/2, [i, j, k ,l]) 
                        ########################################
                        qc.h([i, j]) 
                        qc.rx(pi/2, [k ,l]) 
                        quartic_exp(qc=qc, theta=-(1/steps)*theta_3[index_3]/4, qubits=[i, j, k , l])
                        qc.h([i, j]) 
                        qc.rx(-pi/2, [k ,l]) 

                        qc.h([k, l]) 
                        qc.rx(pi/2, [i ,j]) 
                        quartic_exp(qc=qc, theta=-(1/steps)*theta_3[index_3]/4, qubits=[i, j, k , l])
                        qc.h([k, l]) 
                        qc.rx(-pi/2, [i ,j])
                        ########################################
                        qc.h([i, k]) 
                        qc.rx(pi/2, [j ,l]) 
                        quartic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/4, qubits=[i, j, k , l])
                        qc.h([i, k]) 
                        qc.rx(-pi/2, [j ,l]) 

                        qc.h([i, l]) 
                        qc.rx(pi/2, [k ,j]) 
                        quartic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/4, qubits=[i, j, k , l])
                        qc.h([i, l]) 
                        qc.rx(-pi/2, [k ,j])       
                        #######################################
                        qc.h([k, j]) 
                        qc.rx(pi/2, [i ,l]) 
                        quartic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/4, qubits=[i, j, k , l])
                        qc.h([k, j]) 
                        qc.rx(-pi/2, [i ,l]) 

                        qc.h([l, j]) 
                        qc.rx(pi/2, [k ,i]) 
                        quartic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/4, qubits=[i, j, k , l])
                        qc.h([l, j]) 
                        qc.rx(-pi/2, [k ,i]) 
                        
                        index_3 = index_3 + 1

                        #############################################
                        # a^dagger_l a^dagger_k a_j a_i - h.c
                        qc.h([i, j, k]) 
                        qc.rx(pi/2, [l]) 
                        quartic_exp(qc=qc, theta=-(1/steps)*theta_4[index_4]/4, qubits=[i, j, k , l])
                        qc.h([i, j, k]) 
                        qc.rx(-pi/2, [l])

                        qc.h([i, j, l]) 
                        qc.rx(pi/2, [k]) 
                        quartic_exp(qc=qc, theta=-(1/steps)*theta_4[index_4]/4, qubits=[i, j, k , l])
                        qc.h([i, j, l]) 
                        qc.rx(-pi/2, [k])

                        qc.h([i, l, k]) 
                        qc.rx(pi/2, [j]) 
                        quartic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/4, qubits=[i, j, k , l])
                        qc.h([i, l, k]) 
                        qc.rx(-pi/2, [j]) 

                        qc.h([l, j, k]) 
                        qc.rx(pi/2, [i]) 
                        quartic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/4, qubits=[i, j, k , l])
                        qc.h([l, j, k]) 
                        qc.rx(-pi/2, [i]) 
                        #########################################
                        qc.h([l]) 
                        qc.rx(pi/2, [i, j, k]) 
                        quartic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/4, qubits=[i, j, k , l])
                        qc.h([l]) 
                        qc.rx(-pi/2, [i, j, k])

                        qc.h([i, j, l]) 
                        qc.rx(pi/2, [k]) 
                        quartic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/4, qubits=[i, j, k , l])
                        qc.h([i, j, l]) 
                        qc.rx(-pi/2, [k])

                        qc.h([i, l, k]) 
                        qc.rx(pi/2, [j]) 
                        quartic_exp(qc=qc, theta=-(1/steps)*theta_4[index_4]/4, qubits=[i, j, k , l])
                        qc.h([i, l, k]) 
                        qc.rx(-pi/2, [j]) 

                        qc.h([l, j, k]) 
                        qc.rx(pi/2, [i]) 
                        quartic_exp(qc=qc, theta=-(1/steps)*theta_4[index_4]/4, qubits=[i, j, k , l])
                        qc.h([l, j, k]) 
                        qc.rx(-pi/2, [i]) 

                        index_4 = index_4 + 1

    return qc

##########################################
########## UNDER CONSTRUCTION ############
##########################################
def mix_gauges(num_qubits):
    phi = ParameterVector('phi', length=num_qubits)
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(phi[i], i)
    return qc


def PGSU(num_qubits, fermions_qubits, det=1, steps=1, param_name='th'):
    m = len(fermions_qubits)
    # defining the parameters used in this ansatz. 
    length = 0 
    for i in range(m): 
        for _ in range(i+1, num_qubits): 
            length = length +1

    theta_3 = ParameterVector(f'{param_name}_3', length=length)
    theta_4 = ParameterVector(f'{param_name}_4', length=length)

    qc = QuantumCircuit(num_qubits)
    for _ in range(steps):
        gauge_qubits = [*range(m,num_qubits)]
        fermions_qubits_p = fermions_qubits.copy()
        index_3 = 0
        index_4 = 0
        # active_qubits_p = active_qubits.copy()
        for i in fermions_qubits:
            fermions_qubits_p.pop(0)
            for j in fermions_qubits_p: 
                #####################################################
                # a_i a_j + a^dagger_j a^dagger_i 
                qc.h([i, j]) 
                quadratic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.h([i, j]) 

                qc.rx(pi/2, [i, j])
                quadratic_exp(qc=qc, theta=-(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.rx(-pi/2, [i, j])

                index_3 = index_3 + 1  
                #####################################################
                # a_i a_j - a^dagger_j a^dagger_i 
                qc.h([i]) 
                qc.rx(pi/2, [j])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/2, qubits=[i,j])
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/2, qubits=[i,j])
                qc.h([j]) 
                qc.rx(-pi/2, [i])

                index_4 = index_4 + 1  

            for j in gauge_qubits: 
                #####################################################
                # g^dagger_j a_i + a^dagger_i g^_j 
                qc.h([i, j]) 
                quadratic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.h([i, j]) 

                qc.rx(pi/2, [i, j])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.rx(-pi/2, [i, j])

                index_3 = index_3 + 1  
                #####################################################
                # g^dagger_j a_i - a^dagger_i g^_j 
                qc.h([i]) 
                qc.rx(pi/2, [j])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/2, qubits=[i,j])
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                quadratic_exp(qc=qc, theta=-(1/steps)*theta_4[index_4]/2, qubits=[i,j])
                qc.h([j]) 
                qc.rx(-pi/2, [i])

                index_4 = index_4 + 1  


    if det == -1:
        qc.x([0])
    return qc



def PGBSDU(num_qubits, fermions_qubits, det=1, steps=1):
    m = len(fermions_qubits)
    active_qubits = [*range(num_qubits)]
    fermions_qubits_p = fermions_qubits.copy()
    gauge_qubits = [*range(m, num_qubits)]
    gauge_qubits_p = [*range(m, num_qubits)]

    # defining the parameters used in this ansatz. 
    length_s = 0 
    length_d = 0 

    for i in range(m): 
        for _ in range(i+1, num_qubits): 
            length_s = length_s +1

    for i in range(m): 
        for _ in range(i+1, m): 
            for j in range(m, num_qubits): 
                for _ in range(j+1, num_qubits): 
                    length_d = length_d +1

    theta_3 = ParameterVector('th_3', length=length_s)
    theta_4 = ParameterVector('th_4', length=length_s)
    phi_1 = ParameterVector('phi_1', length=length_d)
    phi_2 = ParameterVector('phi_2', length=length_d)
    # phi_3 = ParameterVector('phi_3', length=length_d)
    # phi_4 = ParameterVector('phi_4', length=length_d)

    qc = QuantumCircuit(num_qubits)
    for _ in range(steps):
        index_s_3 = 0
        index_s_4 = 0
        index_d_1 = 0
        index_d_2 = 0
        # index_d_3 = 0
        # index_d_4 = 0
        for i in fermions_qubits:
            active_qubits.pop(0)
            fermions_qubits_p.pop(0)
            for j in active_qubits: 
                #####################################################
                # a_i a_j + a^dagger_j a^dagger_i 
                qc.h([i, j]) 
                quadratic_exp(qc=qc, theta=(1/steps)*theta_3[index_s_3]/2, qubits=[i,j])
                qc.h([i, j]) 

                qc.rx(pi/2, [i, j])
                quadratic_exp(qc=qc, theta=-(1/steps)*theta_3[index_s_3]/2, qubits=[i,j])
                qc.rx(-pi/2, [i, j])

                index_3 = index_3 + 1  
                #####################################################
                # a_i a_j - a^dagger_j a^dagger_i 
                qc.h([i]) 
                qc.rx(pi/2, [j])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_s_4]/2, qubits=[i,j])
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_s_4]/2, qubits=[i,j])
                qc.h([j]) 
                qc.rx(-pi/2, [i])

                index_4 = index_4 + 1  
            for j in fermions_qubits_p:
                for ip in gauge_qubits: 
                    gauge_qubits_p.pop(0)
                    for jp in gauge_qubits_p: 
                        #####################################################
                        # (a_ip a_jp + a^dagger_jp a^dagger_ip) (a_i a_j + a^dagger_j a^dagger_i) 
                        qc.h([i, j, ip, jp]) 
                        quartic_exp(qc=qc, theta=(1/steps)*phi_1[index_d_1]/2, qubits=[i, j, ip, jp])
                        qc.h([i, j, ip, jp]) 

                        qc.rx(pi/2, [i, j, ip, jp])
                        quartic_exp(qc=qc, theta=(1/steps)*phi_1[index_d_1]/2, qubits=[i, j, ip, jp])
                        qc.rx(-pi/2, [i, j, ip, jp])

                        index_d_1 = index_d_1 + 1  

                        # (a_ip a_jp - a^dagger_jp a^dagger_ip) (a_i a_j - a^dagger_j a^dagger_i) 
                        qc.h([i, ip]) 
                        qc.rx(pi/2, [j, jp])
                        quartic_exp(qc=qc, theta=(1/steps)*phi_2[index_d_2]/2, qubits=[i, j, ip, jp])
                        qc.h([i, ip]) 
                        qc.rx(-pi/2, [j, jp])

                        qc.h([j, jp]) 
                        qc.rx(pi/2, [i, ip])
                        quartic_exp(qc=qc, theta=(1/steps)*phi_2[index_d_2]/2, qubits=[i, j, ip, jp])
                        qc.h([j, jp]) 
                        qc.rx(-pi/2, [i, ip])

                        index_d_2 = index_d_2 + 1
    
    if det == -1:
        qc.x([0])
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
                quadratic_exp(qc=qc, theta=(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.h([i, j]) 

                qc.rx(pi/2, [i, j])
                quadratic_exp(qc=qc, theta=-(1/steps)*theta_3[index_3]/2, qubits=[i,j])
                qc.rx(-pi/2, [i, j])

                index_3 = index_3 + 1  
                #####################################################
                # a_i a_j - a^dagger_j a^dagger_i 
                qc.h([i]) 
                qc.rx(pi/2, [j])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/2, qubits=[i,j])
                qc.h([i]) 
                qc.rx(-pi/2, [j])

                qc.h([j]) 
                qc.rx(pi/2, [i])
                quadratic_exp(qc=qc, theta=(1/steps)*theta_4[index_4]/2, qubits=[i,j])
                qc.h([j]) 
                qc.rx(-pi/2, [i])

                index_4 = index_4 + 1  
    
    if det == -1:
        qc.x([0])
    return qc