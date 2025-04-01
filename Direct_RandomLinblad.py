import numpy as np
import scipy as sp
import time
from math import factorial
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import seaborn as sns

################################################
# Parameters for plotting things
################################################

rcParams['text.usetex'] = True

# Default matplotlib colour scheme
# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']

# Custom colour scheme
# colors = ["skyblue","crimson","mediumorchid","springgreen","orange"]
colors = ["#78AAC5","#AA2228","#1446A0","#F18805","#81F499"]


################################################
# Seed for randomness
################################################

rng_states = np.random.default_rng(seed=897)
rng_linblad = np.random.default_rng(seed=654)
rng_MCMC = np.random.default_rng(seed=321)

output_file_name = "RandLin_987_654_321_150"

################################################
# Paulis and other classic operators
################################################

X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1*1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
S = np.array([[1,0],[0,1j]])
Id2 = np.array([[1,0],[0,1]])
Id4 = np.identity(4)
H = 1/np.sqrt(2)*np.array([[1,1],[1,-1]])
SWAP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])

################################################
# Kroenecker Powers
################################################

def kron_pow(x,n):
    D = np.shape(x)[0]

    if (n == 0):
        return np.array([1])
    elif (n == 1):
        return x
    else:
        out = np.kron(x,x)
        for i in range(2,n):
            out = np.kron(out,x)

        return out

################################################
# Qubits and projectors
################################################

# Computationnal Basis States
Ket0 = np.array([[1],[0]])
Ket1 = np.array([[0],[1]])

# Qubit from array
def Qn(arr):
    d = len(arr)

    if arr[0] == 0:
        state = Ket0
    else:
        state = Ket1

    if d >= 2:
        for i in range(1,d):
            if arr[i] == 0:
                state = np.kron(state,Ket0)
            else:
                state = np.kron(state,Ket1)

    return state

# Qubit from string
def QnS(bstring):
    d = len(bstring)

    if bstring[0] == '0':
        state = Ket0
    else:
        state = Ket1

    if d >= 2:
        for i in range(1,d):
            if bstring[i] == '0':
                state = np.kron(state,Ket0)
            else:
                state = np.kron(state,Ket1)

    return state

# Projector for pure state
def ProjQ(qubit):
    return np.outer(qubit,np.conj(qubit))

# Cross-term projector
def ProjQX(qubit1,qubit2):
    return np.outer(qubit1,np.conj(qubit2))

################################################
# Fidelity between a pure and mixed state
################################################

def FidelityPM(state,rho):
    return np.real(np.trace(np.matmul(ProjQ(state),rho)))

################################################
# Fidelity between two mixed states
################################################

def FidelityMM(rho1,rho2):
    sqrtrho = sp.linalg.sqrtm(rho1)
    return np.abs(np.trace( sp.linalg.sqrtm( sqrtrho @ rho2 @ sqrtrho ) ))**2

################################################
# Vecor Operator Correspondance
################################################

def OpToVec(op):
    return op.flatten()

def VecToOp(vec,N1,N2):
    return np.reshape(vec,(N1,N2))

################################################
# Partial Trace
################################################

# For tensor product structure of Environment x Probe
# Partial Trace on the environment
def PartialTrace(X,dE,dS):
    temp = X.reshape((dE,dS,dE,dS))
    return np.trace(temp,axis1=0,axis2=2)

# For tensor product structure of Environment x Probe
# Partial Trace on the system
def PartialTrace2(X,dE,dS):
    temp = X.reshape((dE,dS,dE,dS))
    return np.trace(temp,axis1=1,axis2=3)

################################################
# Measurement Residual States in
# the vectorized representation
################################################

# Residual of outcome n on first subsystem
def MeasResidualVec(X,n,dE,dS):
    output = np.zeros(dS*dS,dtype=np.complex128)
    for i in range(dS):
        for j in range(dS):
            output[i*dS+j] = X[n*dE*dS*dS + i*dE*dS + n*dS + j]

    return output

# Residual of outcome n on second subsystem
def MeasResidualVec2(X,n,dE,dS):
    output = np.zeros(dE*dE,dtype=np.complex128)
    for i in range(dE):
        for j in range(dE):
            output[i*dE+j] = X[i*dE*dS*dS + n*dE*dS + j*dS + n]

    return output

################################################
# Kronecker Product in vec Representation
################################################

def KronVec(X1,X2):
    dE = int(np.sqrt(X1.size))
    dS = int(np.sqrt(X2.size))

    output = np.zeros(dS*dS*dE*dE,dtype=np.complex128)

    for i in range(dE):
        for j in range(dE):
            for m in range(dS):
                for n in range(dS):
                    output[i*dE*dS*dS + m*dE*dS + j*dS + n] = X1[i*dE + j] * X2[m*dS + n]

    return output

################################################
# Quantum Fisher Information
################################################

# Quantum Fisher Information
def QuantumFisherInformation_SingleQubit(rho,drho):

    eigs, vecarr = np.linalg.eig(rho)
    eigs = np.real(eigs)
    eigvecs = [
        np.array([vecarr[:,0]]).T,
        np.array([vecarr[:,1]]).T
    ]

    QFI = 0
    for i in range(0,2):
        for j in range(0,2):
            denom = eigs[j] + eigs[i]
            num = np.matmul(np.conj(eigvecs[i].T),np.matmul(drho,eigvecs[j])) / (np.linalg.norm(eigvecs[i])*np.linalg.norm(eigvecs[j]))

            # NOTE 1e-12 is a tolerance here to avoid numerical errors
            # TODO Make the tolerance a parameter of the function
            if np.abs(denom) >= 1e-12 :
                QFI += 2*np.abs(np.trace(num))**2 / denom

    return QFI

################################################
# Sine distribution for Haar random unitaries
# (See Olivia di Matteo's Pennylane
#  Haar measure sampling tutorial)
# https://pennylane.ai/qml/demos/tutorial_haar_measure/
################################################

class sin_prob_dist(sp.stats.rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)

sin_sampler = sin_prob_dist(a=0, b=np.pi, seed=rng_states)

# Generate a Haar random pure state
# Outputs state vector
def HaarRandomPureState():
    phi, omega = 2 * np.pi * rng_states.uniform(size=2)
    theta = sin_sampler.rvs(size=1)

    Haar_state = (np.exp(-1j*(phi+omega)/2)*np.cos(theta/2))*Ket0 + np.exp(-1j*(phi-omega)/2)*np.sin(theta/2)*Ket1

    return Haar_state

# Generate a set of Haar random pure states
# Outputs density matrix
def HaarRandomStates(samples):
    states = []
    for i in range(samples):
        states.append(ProjQ(HaarRandomPureState()))

    return states

################################################
# Derivative for matrix exponential with
# non-commuting matrices
################################################

# Derivative with respect to omega of an exponential of the form
# exp(omega*A + B) where A and B do not commute
def DiffPropagator(omega,t,L_signal,L_rest,truncation):

    derivative = np.zeros(L_signal.shape,dtype='complex128')

    # Index of the Taylor expansion
    for k in range(1,truncation+1):
        # Index of derivative product-rule
        for i in range(0,k):
            term = np.linalg.matrix_power(omega*t*L_signal + t*L_rest,i)
            term = (t*L_signal) @ term
            term = term @ np.linalg.matrix_power(omega*t*L_signal + t*L_rest,k-i-1)

            derivative += 1/factorial(k) * term

    return derivative

#######################################################################
# Generate N by N Random Ginibre Matrix
#######################################################################

def ginibre(N):

    Z1 = rng_linblad.normal(0.0,1.0,N**2)
    Z2 = rng_linblad.normal(0.0,1.0,N**2)

    Z = 1/np.sqrt(2) * ( np.reshape(Z1,(N,N)) + 1j*np.reshape(Z2,(N,N)) )

    return Z

################################################
# Convert integer to array of bits
################################################

def bitfield(n):
    return [1 if digit=='1' else 0 for digit in bin(n)[2:]]

################################################
# Quantum Fisher Information for a
# General Prepare and Measure Strategy
# calculated exactly
################################################

def QFI_Trajectory(rho_E,state_S,basis_op,meas_op,H_signal,H_SE,jump_operators,omega,delta_t,N):

    ################################################
    # rho_E = initial environment state
    # rho_S = probe state to be prepared
    # meas_op = unitary prior to computationnal
    #           basis measurement
    # H_signal = signal Hamiltonian
    # Linbladian_SG = semi-group generator for
    #                 the Linbladian
    ################################################

    # Initial State
    rho_S = ProjQ(state_S)
    vec_S = OpToVec(rho_S)
    ini_state = np.kron(rho_E,rho_S)

    # Projectors onto the + and - States
    eigs, vecarr = np.linalg.eig(meas_op)
    eigs = np.real(eigs)
    eigvecs = [
        np.array([vecarr[:,0]]).T,
        np.array([vecarr[:,1]]).T
    ]

    # Projectors onto the eigenstates of meas. op.
    # NOTE Assigment of +- 1 to the outcomes is arbitary
    ProjPlus = ProjQ(eigvecs[0])
    ProjMinus = ProjQ(eigvecs[1])

    # Function for Linbaldian SG
    Linbladian_SG = np.zeros((16,16),dtype='complex128')
    for jump_op in jump_operators:
        Linbladian_SG += np.kron(jump_op, np.conj(jump_op))
        LdagL = np.conj(jump_op.T) @ jump_op
        Linbladian_SG -= 0.5*(np.kron(LdagL,Id4) + np.kron(Id4,LdagL.T))

    # Time Evolution super-operators
    H_SE_SG = np.kron(-1j*H_SE,Id4)+np.kron(Id4,1j*np.conj(H_SE))
    H_signal_SG = np.kron(-1j*H_signal,Id4)+np.kron(Id4,1j*np.conj(H_signal))

    time_evolution_SOP = sp.linalg.expm(delta_t*(H_SE_SG + omega*H_signal_SG + Linbladian_SG))
    pre_meas_SOP = np.kron(np.kron(Id2,basis_op),np.kron(Id2,np.conj(basis_op)))

    factor_diff = DiffPropagator(omega,delta_t,H_signal_SG,H_SE_SG+Linbladian_SG,100)

    # Time Evolution
    QFIs = np.zeros(N)
    probs = np.zeros(2**N)

    # Loop over all combinations of measurement outcomes
    # for the prep-and-measure trajectory
    for i in range(2**N):

        path = bitfield(i)
        path = np.pad(path,(N-len(path),0),'constant',constant_values=(0,0))

        ini_state_vec = OpToVec(ini_state)
        ini_state_diff = np.matmul(time_evolution_SOP,ini_state_vec)

        for n in range(N):
            # Time Evolution
            time_evolved_state = np.matmul(time_evolution_SOP,ini_state_vec)
            time_evolved_diff = np.matmul(time_evolution_SOP,ini_state_diff)

            if n >= 1:
                diff_density_matrix = factor_diff @ ini_state_vec + time_evolved_diff
            else:
                diff_density_matrix = factor_diff @ ini_state_vec

            # Quantum Fisher Information
            reduced_dm = PartialTrace(VecToOp(time_evolved_state,4,4),2,2)
            diff_reduced_dm = PartialTrace(VecToOp(diff_density_matrix,4,4),2,2)

            if path[n] == 0:
                reduced_dm = np.matmul(ProjPlus,np.matmul(reduced_dm,ProjPlus))
                diff_reduced_dm = np.matmul(ProjPlus,np.matmul(diff_reduced_dm,ProjPlus))
            else:
                reduced_dm = np.matmul(ProjMinus,np.matmul(reduced_dm,ProjMinus))
                diff_reduced_dm = np.matmul(ProjMinus,np.matmul(diff_reduced_dm,ProjMinus))

            # Calculate the QFI of the state
            # which corresponds to the classical FI of the measurement
            # since we project into the basis
            # NOTE The FI could be calculated more efficiently here by calculating the CFI directly,
            # however the bottleneck for the exact calculation is the looping over 2^N rather
            # than the diagonalization of the 4 by 4 matrix
            block_qfi = QuantumFisherInformation_SingleQubit(reduced_dm,diff_reduced_dm)

            QFIs[n] += block_qfi / (2**(N-n-1))

            # Prep. for next run of loop
            if path[n] == 0:
                density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,time_evolved_state),0,2,2)
                diff_density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,diff_density_matrix),0,2,2)
            else:
                density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,time_evolved_state),1,2,2)
                diff_density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,diff_density_matrix),1,2,2)

            ini_state_vec = KronVec(density_matrix,vec_S)
            ini_state_diff = KronVec(diff_density_matrix,vec_S)

        probs[i] = np.abs(np.trace(VecToOp(density_matrix,2,2)))

    return QFIs, probs

################################################
# Quantum Fisher Information for a
# General Prepare and Measure Strategy
# using statistical sampling
################################################

def QFI_Trajectory_direct(rho_E,state_S,basis_op,meas_op,H_signal,H_SE,jump_operators,omega,delta_t,N,samples):

    ################################################
    # rho_E = initial environment state
    # rho_S = probe state to be prepared
    # meas_op = unitary prior to computationnal
    #           basis measurement
    # H_signal = signal Hamiltonian
    # Linbladian_SG = semi-group generator for
    #                 the Linbladian
    ################################################

    # Initial State
    rho_S = ProjQ(state_S)
    vec_S = OpToVec(rho_S)
    ini_state = np.kron(rho_E,rho_S)

    # Projectors onto the + and - States
    eigs, vecarr = np.linalg.eig(meas_op)
    eigs = np.real(eigs)
    eigvecs = [
        np.array([vecarr[:,0]]).T,
        np.array([vecarr[:,1]]).T
    ]

    # Projectors onto the eigenstates of meas. op.
    # NOTE Assigment of +- 1 to the outcomes is arbitary
    ProjPlus = ProjQ(eigvecs[0])
    ProjMinus = ProjQ(eigvecs[1])

    # Function for Linbaldian SG
    Linbladian_SG = np.zeros((16,16),dtype='complex128')
    for jump_op in jump_operators:
        Linbladian_SG += np.kron(jump_op, np.conj(jump_op))
        LdagL = np.conj(jump_op.T) @ jump_op
        Linbladian_SG -= 0.5*(np.kron(LdagL,Id4) + np.kron(Id4,LdagL.T))

    # Time Evolution super-operators
    H_SE_SG = np.kron(-1j*H_SE,Id4)+np.kron(Id4,1j*np.conj(H_SE))
    H_signal_SG = np.kron(-1j*H_signal,Id4)+np.kron(Id4,1j*np.conj(H_signal))

    time_evolution_SOP = sp.linalg.expm(delta_t*(H_SE_SG + omega*H_signal_SG + Linbladian_SG))
    pre_meas_SOP = np.kron(np.kron(Id2,basis_op),np.kron(Id2,np.conj(basis_op)))

    factor_diff = DiffPropagator(omega,delta_t,H_signal_SG,H_SE_SG+Linbladian_SG,100)

    # Time Evolution
    QFIs = np.zeros(N)

    # QFI of Sample
    def QFI_of_Path():

        temp_QFIs = np.zeros(N)

        ini_state_vec = OpToVec(ini_state)
        ini_state_diff = np.matmul(time_evolution_SOP,ini_state_vec)

        for n in range(N):
            # Time Evolution
            time_evolved_state = np.matmul(time_evolution_SOP,ini_state_vec)
            time_evolved_diff = np.matmul(time_evolution_SOP,ini_state_diff)

            if n >= 1:
                diff_density_matrix = factor_diff @ ini_state_vec + time_evolved_diff
            else:
                diff_density_matrix = factor_diff @ ini_state_vec

            # Quantum Fisher Information
            reduced_dm = PartialTrace(VecToOp(time_evolved_state,4,4),2,2)
            diff_reduced_dm = PartialTrace(VecToOp(diff_density_matrix,4,4),2,2)

            # Direct Sampling
            reduced_dm_normalized = reduced_dm/np.abs(np.trace(reduced_dm))
            prob_0 = np.abs(np.trace(np.matmul(ProjPlus,np.matmul(reduced_dm_normalized,ProjPlus))))
            alpha = rng_states.uniform(size=1)
            if alpha < prob_0:
                choice = 0
            else:
                choice = 1

            # Next step
            if choice == 0:
                reduced_dm = np.matmul(ProjPlus,np.matmul(reduced_dm,ProjPlus))
                diff_reduced_dm = np.matmul(ProjPlus,np.matmul(diff_reduced_dm,ProjPlus))
            else:
                reduced_dm = np.matmul(ProjMinus,np.matmul(reduced_dm,ProjMinus))
                diff_reduced_dm = np.matmul(ProjMinus,np.matmul(diff_reduced_dm,ProjMinus))

            # Calculate the classical FI for the measurement
            traj_prob = np.abs(np.trace(reduced_dm))
            diff_traj_prob = np.abs(np.trace(diff_reduced_dm))
            block_qfi = (diff_traj_prob / traj_prob)**2

            temp_QFIs[n] += block_qfi

            # Prep. for next run of loop
            if choice == 0:
                density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,time_evolved_state),0,2,2)
                diff_density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,diff_density_matrix),0,2,2)
            else:
                density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,time_evolved_state),1,2,2)
                diff_density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,diff_density_matrix),1,2,2)

            ini_state_vec = KronVec(density_matrix,vec_S)
            ini_state_diff = KronVec(diff_density_matrix,vec_S)

        # prob = np.abs(np.trace(VecToOp(density_matrix,2,2)))
        return temp_QFIs

    first = True
    for i in range(samples):
        temp_QFIs = QFI_of_Path()
        QFIs = QFIs + temp_QFIs

        if first:
            max_scores = QFIs
            first = False
        else:
            max_scores = np.maximum(max_scores,temp_QFIs)

    return QFIs/(samples), max_scores

################################################
# Quantum Fisher Information for a
# General Prepare and Measure Strategy
# using statistical sampling
################################################

def QFI_Trajectory_direct(rho_E,state_S,basis_op,meas_op,H_signal,H_SE,jump_operators,omega,delta_t,N,samples):

    ################################################
    # rho_E = initial environment state
    # rho_S = probe state to be prepared
    # meas_op = unitary prior to computationnal
    #           basis measurement
    # H_signal = signal Hamiltonian
    # Linbladian_SG = semi-group generator for
    #                 the Linbladian
    ################################################

    # Initial State
    rho_S = ProjQ(state_S)
    vec_S = OpToVec(rho_S)
    ini_state = np.kron(rho_E,rho_S)

    # Projectors onto the + and - States
    eigs, vecarr = np.linalg.eig(meas_op)
    eigs = np.real(eigs)
    eigvecs = [
        np.array([vecarr[:,0]]).T,
        np.array([vecarr[:,1]]).T
    ]

    # Projectors onto the eigenstates of meas. op.
    # NOTE Assigment of +- 1 to the outcomes is arbitary
    ProjPlus = ProjQ(eigvecs[0])
    ProjMinus = ProjQ(eigvecs[1])

    # Function for Linbaldian SG
    Linbladian_SG = np.zeros((16,16),dtype='complex128')
    for jump_op in jump_operators:
        Linbladian_SG += np.kron(jump_op, np.conj(jump_op))
        LdagL = np.conj(jump_op.T) @ jump_op
        Linbladian_SG -= 0.5*(np.kron(LdagL,Id4) + np.kron(Id4,LdagL.T))

    # Time Evolution super-operators
    H_SE_SG = np.kron(-1j*H_SE,Id4)+np.kron(Id4,1j*np.conj(H_SE))
    H_signal_SG = np.kron(-1j*H_signal,Id4)+np.kron(Id4,1j*np.conj(H_signal))

    time_evolution_SOP = sp.linalg.expm(delta_t*(H_SE_SG + omega*H_signal_SG + Linbladian_SG))
    pre_meas_SOP = np.kron(np.kron(Id2,basis_op),np.kron(Id2,np.conj(basis_op)))

    factor_diff = DiffPropagator(omega,delta_t,H_signal_SG,H_SE_SG+Linbladian_SG,100)

    # Time Evolution
    QFIs = np.zeros(N)

    # QFI of Sample
    def QFI_of_Path():

        temp_QFIs = np.zeros(N)

        ini_state_vec = OpToVec(ini_state)
        ini_state_diff = np.matmul(time_evolution_SOP,ini_state_vec)

        for n in range(N):
            # Time Evolution
            time_evolved_state = np.matmul(time_evolution_SOP,ini_state_vec)
            time_evolved_diff = np.matmul(time_evolution_SOP,ini_state_diff)

            if n >= 1:
                diff_density_matrix = factor_diff @ ini_state_vec + time_evolved_diff
            else:
                diff_density_matrix = factor_diff @ ini_state_vec

            # Quantum Fisher Information
            reduced_dm = PartialTrace(VecToOp(time_evolved_state,4,4),2,2)
            diff_reduced_dm = PartialTrace(VecToOp(diff_density_matrix,4,4),2,2)

            # Direct Sampling
            reduced_dm_normalized = reduced_dm/np.abs(np.trace(reduced_dm))
            prob_0 = np.abs(np.trace(np.matmul(ProjPlus,np.matmul(reduced_dm_normalized,ProjPlus))))
            alpha = rng_states.uniform(size=1)
            if alpha < prob_0:
                choice = 0
            else:
                choice = 1

            # Next step
            if choice == 0:
                reduced_dm = np.matmul(ProjPlus,np.matmul(reduced_dm,ProjPlus))
                diff_reduced_dm = np.matmul(ProjPlus,np.matmul(diff_reduced_dm,ProjPlus))
            else:
                reduced_dm = np.matmul(ProjMinus,np.matmul(reduced_dm,ProjMinus))
                diff_reduced_dm = np.matmul(ProjMinus,np.matmul(diff_reduced_dm,ProjMinus))

            # Calculate the classical FI for the measurement
            traj_prob = np.abs(np.trace(reduced_dm))
            diff_traj_prob = np.abs(np.trace(diff_reduced_dm))
            block_qfi = (diff_traj_prob / traj_prob)**2

            temp_QFIs[n] += block_qfi

            # Prep. for next run of loop
            if choice == 0:
                density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,time_evolved_state),0,2,2)
                diff_density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,diff_density_matrix),0,2,2)
            else:
                density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,time_evolved_state),1,2,2)
                diff_density_matrix = MeasResidualVec2(np.matmul(pre_meas_SOP,diff_density_matrix),1,2,2)

            ini_state_vec = KronVec(density_matrix,vec_S)
            ini_state_diff = KronVec(diff_density_matrix,vec_S)

        # prob = np.abs(np.trace(VecToOp(density_matrix,2,2)))
        return temp_QFIs

    first = True
    for i in range(samples):
        temp_QFIs = QFI_of_Path()
        QFIs = QFIs + temp_QFIs

        if first:
            max_scores = QFIs
            first = False
        else:
            max_scores = np.maximum(max_scores,temp_QFIs)

    return QFIs/(samples), max_scores

################################################
# Heisenberg Interaction
# System only dephazing
################################################

# Calculate the QFIs for the Heisenberg type interaction
def QFISingleQubitHeisenberg(rho_E,gamma,omega,theta,delta_t,N_exact,N_direct,samples):

    state_S = 1/np.sqrt(2)*(Ket0 + np.exp(-1j*theta)*Ket1)

    # Measurement operator and basis chosen
    meas_op = X
    basis_op = H

    H_signal = np.kron(Id2,Z)
    H_SE = gamma*(np.kron(Z,Z) + np.kron(X,X) + np.kron(Y,Y))
    jump_operators = [np.kron(Id2,Z)]

    start_time = time.time()
    QFIS, probs = QFI_Trajectory(rho_E,state_S,basis_op,meas_op,H_signal,H_SE,jump_operators,omega,delta_t,N_exact)
    print(time.time()-start_time," seconds for exact calculation")
    start_time = time.time()
    QFIS_direct, max_scores_direct = QFI_Trajectory_direct(rho_E,state_S,basis_op,meas_op,H_signal,H_SE,jump_operators,omega,delta_t,N_direct,samples)
    print(time.time()-start_time," seconds for direct calculation")

    return QFIS, probs, QFIS_direct, max_scores_direct

################################################
# Quantum Fisher Information for a
# General Prepare and Measure Strategy
# using Direct Sampling
################################################

def QFISingleQubitRandomLinblad(delta_t,omega,theta,r_L,N_exact,N_direct,samples):

    # Random Environment State and initial probe state
    rho_E = HaarRandomStates(1)
    state_S = 1/np.sqrt(2)*(Ket0 + np.exp(-1j*theta)*Ket1)

    # Measurement operator and basis chosen
    meas_op = X
    basis_op = H

    H_signal = np.kron(Id2,Z)
    A_SE = ginibre(4)
    H_SE = 0.5*(A_SE + np.conj(A_SE.T))

    jump_operators = [ginibre(4) for i in range(r_L)]

    start_time = time.time()
    QFIs, probs = QFI_Trajectory(rho_E,state_S,basis_op,meas_op,H_signal,H_SE,jump_operators,omega,delta_t,N_exact)
    print(time.time()-start_time," seconds for exact calculation")
    start_time = time.time()
    QFIs_direct, max_scores_direct = QFI_Trajectory_direct(rho_E,state_S,basis_op,meas_op,H_signal,H_SE,jump_operators,omega,delta_t,N_direct,samples)
    print(time.time()-start_time," seconds for exact calculation")

    return QFIs, probs, QFIs_direct, max_scores_direct, rho_E, H_SE, jump_operators

# Generate Plot of Random Linblad Dynamics
def PlotQFISingleQubitRandomLinblad(curves,delta_t,omega,theta,r_Ls,N_exact,N_direct,samples,fitting,verbose,error_bars,fig_title):

    Ns_exact = np.array(range(0,N_exact))
    Ns_direct = np.array(range(0,N_direct))

    if fitting:
        rvals = np.zeros(curves)
        slopes = np.zeros(curves)

    for i in range(curves):

        if verbose:
            start_time = time.time()

        QFIs, probs, QFIs_direct, max_scores_direct, rho_E, H_SE, jump_operators = QFISingleQubitRandomLinblad(delta_t,omega,theta,r_Ls[i],N_exact,N_direct,samples)

        if fitting:
            fit = sp.stats.linregress(Ns_direct,QFIs_direct)
            slope = fit.slope
            intercept = fit.intercept
            rval = fit.rvalue

            rvals[i] = rval
            slopes[i] = slope

        # Plotting
        color = colors[int(i % len(colors))]

        if fitting:
            plt.plot([0,N_direct],[intercept,slope*N_direct+intercept],linestyle='dashed',color=color)
        else:
            plt.plot(Ns_exact,QFIs,color=color)
            plt.plot(Ns_direct,QFIs_direct,linestyle='dashed',color=color)

        plt.scatter(Ns_exact,QFIs,color=color)
        if error_bars:
            # TODO Finish this section
            plt.errorbar(Ns_direct,QFIs_direct,yerr=((max_scores_direct*QFIs_direct)/samples),color=color,fmt="*")

        else:
            plt.scatter(Ns_direct,QFIs_direct,marker='*',color=color)

        if verbose:
            print("Configuration completed in ", time.time() - start_time, "seconds")
            if fitting:
                print("Slope = ",slope)
                print("Intercept = ",intercept)
                print("r value = ",rval)

    plt.xlabel(r"$N$")
    plt.ylabel(r"$\mathcal{F}$")
    plt.show()
    # plt.savefig("../Paper Plots/%s.png"%(fig_title),dpi=300)
    plt.clf()

    return slopes, rvals

# Slopes and r coefficients box plots
def SQLStatisticsSingleQubitRandomLinblad(curves,delta_t,omega,theta,r_Ls,N_exact,N_direct,samples,verbose):

    Ns_exact = np.array(range(0,N_exact))
    Ns_direct = np.array(range(0,N_direct))

    rvals = np.zeros(curves)
    slopes = np.zeros(curves)

    for i in range(curves):

        if verbose:
            start_time = time.time()

        QFIs, probs, QFIs_direct, max_scores_direct, rho_E, H_SE, jump_operators = QFISingleQubitRandomLinblad(delta_t,omega,theta,r_Ls[i],N_exact,N_direct,samples)

        fit = sp.stats.linregress(Ns_direct,QFIs_direct)
        slope = fit.slope

        if slope < 1e-7 and verbose:
            print("~~~~~ ZERO QFI ~~~~~")
            print(slope)
            print(rho_E)
            print(H_SE)
            print(jump_operators)

        intercept = fit.intercept
        rval = fit.rvalue

        rvals[i] = rval
        slopes[i] = slope

    if verbose:
        print("~~~~~~~")
        print("Min slope = ",np.amin(slopes))

    # plt.boxplot(rvals,vert=False)
    parts = plt.violinplot(rvals,vert=False)

    # for pc in parts['bodies']:
    #     pc.set_facecolor('#EF233C')
    #     pc.set_edgecolor('#153131')
    #     pc.set_alpha(1)


    plt.xlabel(r"$r_{\rm c}$ Coefficient")
    plt.tick_params(axis='y',left=False,labelleft=False)

    plt.show()
    plt.clf()

    # plt.boxplot(slopes,vert=False)
    plt.violinplot(slopes,vert=False)
    plt.tick_params(axis='y',left=False,labelleft=False)
    plt.xlim(left = 0)
    plt.xlabel(r"Slope")

    plt.show()
    plt.clf()

# Calculate r coefficients and slopes for the SQL data
def SQLDataSingleQubitRandomLinblad(curves,steps,delta_t,omega,theta,r_Ls,N_exact,N_direct,samples,file_name):

    Ns_exact = np.array(range(0,N_exact))
    Ns_direct = np.array(range(0,N_direct))

    rvals = np.zeros(curves)
    slopes = np.zeros(curves)

    for i in range(curves):

        QFIs, probs, QFIs_direct, max_scores_direct, rho_E, H_SE, jump_operators = QFISingleQubitRandomLinblad(delta_t,omega,theta,r_Ls[i],N_exact,N_direct,samples)

        fit = sp.stats.linregress(Ns_direct,QFIs_direct)
        slope = fit.slope

        if slope < 1e-7:
            print("~~~~~ ZERO QFI ~~~~~")
            print(slope)
            print(rho_E)
            print(H_SE)
            print(jump_operators)
            print(i)

        intercept = fit.intercept
        rval = fit.rvalue

        rvals[i] = rval
        slopes[i] = slope

        if i % steps == 0:
            save_arr = np.array([range(curves),rvals,slopes]).T
            np.savetxt("../Paper Data/Box Plots Direct/%s.csv"%(file_name), save_arr, delimiter=",")

        print("Curve ",i," completed!")

    save_arr = np.array([range(curves),rvals,slopes]).T
    np.savetxt("../Paper Data/Box Plots Direct/%s.csv"%(file_name), save_arr, delimiter=",")

# Slopes and r coefficients box plots
def PlotStatisticsSingleQubitRandomLinblad(folder):

    directory = os.fsencode(folder)

    first = True

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        add_data = input("Use data from "+filename+"? (y/n)")
        if add_data == "y":
            if first:
                data = np.genfromtxt(folder+filename, delimiter=',')
                first = False
            else:
                file_data = np.genfromtxt(folder+filename, delimiter=',')
                data = np.vstack((data,file_data))

    rvals = data[:,1]
    slopes = data[:,2]

    # Parameters for the boxplots

    r_q1 = np.quantile(rvals,0.25)
    r_q3 = np.quantile(rvals,0.75)
    r_iqr = r_q3 - r_q1

    r_fliers = np.array([x for x in rvals if (x < r_q1 - 1.5*r_iqr or x > r_q3 + 1.5*r_iqr) ])

    slope_q1 = np.quantile(slopes,0.25)
    slope_q3 = np.quantile(slopes,0.75)
    slope_iqr = slope_q3 - slope_q1

    slope_fliers = np.array([x for x in slopes if (x < slope_q1 - 1.5*slope_iqr or x > slope_q3 + 1.5*slope_iqr) ])


    print("Number of data points : ",rvals.size)
    print("Minimum r value : ",np.amin(rvals))
    print("Minimum Slope : ",np.amin(slopes))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    ax1.grid(axis='x')
    ax2.grid(axis='x')

    r_fliers = np.sort(r_fliers)

    sns.boxplot(x=rvals, color = "skyblue",width=0.1,linewidth=2,showfliers = False,ax=ax1)
    sns.stripplot(x=r_fliers[5:],color = "crimson", linewidth = 1,alpha = 0.5,jitter=0.05,ax=ax1)
    sns.stripplot(x=r_fliers[1:5],color = "crimson", linewidth = 1,alpha = 0.5,jitter=0.02,ax=ax1)
    # parts = plt.violinplot(rvals,vert=False)

    sub_axes = plt.axes([.04, .75, .25, .1])
    # sub_axes.set_facecolor('none')

    sns.stripplot(x=r_fliers[0:1],color = "crimson", linewidth = 1,alpha = 0.5,jitter=False,ax=sub_axes)

    # sns.despine(trim = True)

    # plt.xlabel(r"$r$ Coefficient",fontsize=18)
    ax1.set_xlabel(r"$r_{\rm c}$ Coefficient",fontsize=18)

    # plt.ylim(bottom = -0.1)
    # plt.ylim(top = 0.1)
    ax1.set_ylim(top=0.1,bottom=-0.1)

    sub_axes.set_xlim(left=0.981,right=0.983)

    # plt.tick_params(axis='y',left=False,labelleft=False)
    # plt.tick_params(axis='x',labelsize=18)
    ax1.tick_params(axis='y',left=False,labelleft=False)
    ax1.tick_params(axis='x',labelsize=18)

    sub_axes.tick_params(axis='y',left=False,labelleft=False)
    sub_axes.tick_params(axis='x',labelsize=18)
    # sub_axes.xaxis.set_label_position('top')

    ax1.set_title("c)",fontsize=18)

    # plt.tight_layout()

    # plt.savefig("../Paper Plots/Random_r_box.png",dpi=300)
    # plt.show()
    # plt.clf()

    # plt.figure(figsize=(6,2.5))

    ax2.set_title("d)",fontsize=18)


    sns.boxplot(x=slopes, color = "skyblue",width=0.1,linewidth=2,showfliers = False,ax=ax2)
    sns.stripplot(x=slope_fliers,color = "crimson", linewidth = 1,alpha = 0.5,jitter=0.05,ax=ax2)
    # plt.boxplot(slopes,vert=False)
    # plt.violinplot(slopes,vert=False)


    # plt.plot([0,0],[-0.3,0.3],color='grey',alpha=0.5, linestyle='dashed')

    # sns.despine(trim = True)
    # plt.tick_params(axis='y',left=False,labelleft=False)
    # plt.tick_params(axis='x',labelsize=18)
    ax2.tick_params(axis='y',left=False,labelleft=False)
    ax2.tick_params(axis='x',labelsize=18)
    # plt.xlim(left = 0)
    # plt.ylim(bottom = -0.1)
    # plt.ylim(top = 0.1)
    ax2.set_ylim(top=0.1,bottom=-0.1)
    # plt.xlabel(r"Slope",fontsize=18)
    ax2.set_xlabel(r"Slope",fontsize=18)

    plt.tight_layout()

    # plt.savefig("../Paper Plots/Random_slope_box.png",dpi=300)
    plt.savefig("../Paper Plots/Random_boxes_direct_horizontal.png",dpi=300)
    # plt.show()
    plt.clf()

################################################
# Main + Testing
################################################

# slopes, rvals = PlotQFISingleQubitRandomLinblad(5,0.3,0,np.pi/2,np.full(5,2),10,30,2000,25000,True,True,False,"RandLin_42_95_47")

# SQLStatisticsSingleQubitRandomLinblad(20,0.1,0,np.pi/2,np.full(20,2),10,30,2000,25000,True)
# SQLDataSingleQubitRandomLinblad(150,10,0.1,0,np.pi/2,np.full(150,3),10,30,25000,output_file_name)

PlotStatisticsSingleQubitRandomLinblad("../Paper Data/Box Plots Direct/")
