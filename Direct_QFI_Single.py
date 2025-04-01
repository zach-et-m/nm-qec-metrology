import numpy as np
import scipy as sp
import time
from math import factorial
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
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

rng_states = np.random.default_rng(seed=42)
rng_sampler = np.random.default_rng(seed=47)

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
            # rather than being hard-coded in
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
    QFI_variances = np.zeros(N)

    # QFI of Sample
    def QFI_of_Path():

        temp_QFIs = np.zeros(N)
        temp_variances = np.zeros(N)

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
            temp_variances[n] += block_qfi**2

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
        return temp_QFIs, temp_variances

    first = True
    for i in range(samples):
        temp_QFIs, temp_variances = QFI_of_Path()
        QFIs = QFIs + temp_QFIs
        QFI_variances = QFI_variances + temp_variances

        # if first:
        #     max_scores = QFIs
        #     first = False
        # else:
        #     max_scores = np.maximum(max_scores,temp_QFIs)

    print(QFIs/samples)
    # print((QFI_variances - (QFIs/samples)**2 )/(samples**2))
    print((QFI_variances )/(samples**2))

    # return QFIs/(samples), (QFI_variances - (QFIs/samples)**2 )/(samples**2)
    return QFIs/(samples), (QFI_variances)/(samples**2)

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
    QFIS_direct, variances_direct = QFI_Trajectory_direct(rho_E,state_S,basis_op,meas_op,H_signal,H_SE,jump_operators,omega,delta_t,N_direct,samples)
    print(time.time()-start_time," seconds for direct calculation")

    return QFIS, probs, QFIS_direct, variances_direct

# Generate QFI plots for the Heisenberg model
def PlotQFISingleQubitHeisenberg(rho_Es,gammas,omegas,thetas,delta_ts,N_exact,N_direct,verbose,samples,fitting,error_bars,fig_title):

    # plt.figure(figsize=(6,12))
    fig, (ax1, ax2) = plt.subplots(2, 1)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(11, 6)
    fig.set_size_inches(6,11)

    Ns_exact = np.array(range(0,N_exact))
    Ns_direct = np.array(range(0,N_direct))
    legend = False

    counter = 0

    rho_E_text = ""

    # Generate the labels and data for all the combinations of parameters
    for rho_E_idx in range(len(rho_Es)):

        rho_E = rho_Es[rho_E_idx]

        label1 = r""
        label5 = r""
        if len(rho_Es) > 1:
            label1 = r"$\rho_{%s}$, "%(rho_E_idx)
            # label1 = r"$\rho_E=\rho_{%s}$, "%(rho_E_idx)
            label5 = label1

            rho_E_label = r"$\rho_{%s}=\left( \begin{array}{cc} %s & %s \\ %s & %s \end{array} \right) $"%(rho_E_idx,np.round(rho_E[0,0],3),np.round(rho_E[0,1],3),np.round(rho_E[1,0],3),np.round(rho_E[1,1],3))
            rho_E_text = "\n".join((rho_E_text,rho_E_label))

        for gamma in gammas:
            if len(gammas) > 1:
                label2 = label1 + r"$\gamma$=%s, "%(np.round(gamma,5))
                label5 = label2

            for omega in omegas:
                if len(omegas) > 1:
                    label3 = label2 + r"$\omega$=%s, "%(np.round(omega,5))
                    label5 = label3

                for theta in thetas:
                    if len(thetas) > 1:
                        label4 = label3 + r"$\theta$=%s, "%(np.round(theta,5))
                        label5 = label4

                    for delta_t in delta_ts:
                        if len(delta_ts) > 1:
                            label5 = label4 + r"$\Delta t$=%s, "%(np.round(delta_t,5))

                        if verbose:
                            start_time = time.time()

                        # Calculate QFIs for given combination of parameters
                        QFIs, probs, QFIs_direct, variances_direct = QFISingleQubitHeisenberg(rho_E,gamma,omega,theta,delta_t,N_exact,N_direct,samples)

                        # Curve Fit
                        if fitting:
                            fit = sp.stats.linregress(Ns_direct,QFIs_direct)
                            slope = fit.slope
                            intercept = fit.intercept
                            rval = fit.rvalue

                        # Print Details
                        if verbose:
                            print(rho_E)

                        # Plotting
                        color = colors[int(counter % len(colors))]

                        if fitting:
                            ax1.plot([0,N_direct],[intercept,slope*N_direct+intercept],linestyle='dashed',color=color)
                        else:
                            ax1.plot(Ns_exact,QFIs,color=color)
                            ax1.plot(Ns_direct,QFIs_direct,linestyle='dashed',color=color)

                        ax2.plot(Ns_exact,QFIs,color=color)
                        ax2.plot(Ns_exact,QFIs_direct[:N_exact],linestyle='dashed',color=color)

                        ax1.scatter(Ns_exact,QFIs,color=color)
                        ax2.scatter(Ns_exact,QFIs,color=color)
                        if error_bars:
                            # TODO Finish this section
                            ax1.errorbar(Ns_direct,QFIs_direct,yerr=(np.sqrt(variances_direct)),label=label5,color=color,fmt="*")
                            ax2.errorbar(Ns_exact,QFIs_direct[:N_exact],yerr=(np.sqrt(variances_direct[:N_exact])),label=label5,color=color,fmt="*")


                        else:
                            ax1.scatter(Ns_direct,QFIs_direct,marker='*',label=label5,color=color)
                            ax2.scatter(Ns_exact,QFIs_direct[:N_exact],marker='*',label=label5,color=color)

                        counter += 1

                        if verbose:
                            print("Configuration completed in ", time.time() - start_time, "seconds")
                            if fitting:
                                print("Slope = ",slope)
                                print("Intercept = ",intercept)
                                print("r value = ",rval)

                        if label5 != "":
                            legend = True

    title = r"Heisenberg interaction"
    if len(rho_Es) == 1:
        rho_E = rho_Es[0]
        title += r", $\rho_E=\left( \begin{array}{cc} %s & %s \\ %s & %s \end{array} \right) $"%(np.round(rho_E[0,0],3),np.round(rho_E[0,1],3),np.round(rho_E[1,0],3),np.round(rho_E[1,1],3))
    # else:
    #     plt.text(11,0.7,rho_E_text, fontsize=10,
    #     verticalalignment='top')
    if len(gammas) == 1:
        gamma = gammas[0]
        title += r", $\gamma$=%s"%(np.round(gamma,3))
    if len(omegas) == 1:
        omega = omegas[0]
        title += r", $\omega$=%s"%(np.round(omega,3))
    if len(thetas) == 1:
        theta = thetas[0]
        title += r", $\theta$=%s"%(np.round(theta,3))
    if len(delta_ts) == 1:
        delta_t = delta_ts[0]
        title += r", $\Delta t$=%s"%(np.round(delta_t,3))

    plt.tight_layout(pad=4, rect=(0,0,1,1))

    # plt.title(title)
    # plt.xlabel(r"Time step ($t/\Delta t$)")

    ax2.set_xlabel(r"$N$",fontsize=18)
    ax1.set_ylabel(r"$\mathcal{I}(\omega)$",fontsize=18)
    ax2.set_ylabel(r"$\mathcal{I}(\omega)$",fontsize=18)

    ax1.tick_params(axis='x',labelsize=18)
    ax1.tick_params(axis='y',labelsize=18)
    ax2.tick_params(axis='x',labelsize=18)
    ax2.tick_params(axis='y',labelsize=18)

    ax1.set_title("a)",fontsize=18)
    ax2.set_title("b)",fontsize=18)

    if legend:
        ax1.legend(fontsize=18)
    plt.savefig("../Paper Plots/%s.png"%(fig_title),dpi=300)
    plt.clf()

################################################
# Main + Testing
################################################

rho_Es = HaarRandomStates(5)

gammas = np.full(1,1.0)
omegas = np.full(1,0.0)
thetas = np.full(1,np.pi/2)
delta_ts = np.full(1,0.25)

overall_start_time = time.time()

PlotQFISingleQubitHeisenberg(rho_Es,gammas,thetas,thetas,delta_ts,15,60,True,25000,True,True,"Haar_42_47_both_direct")
# PlotQFISingleQubitHeisenberg(rho_Es,gammas,thetas,thetas,delta_ts,5,10,True,2000,2000,True,True,"Haar_42_47_both_direct")
# PlotQFISingleQubitHeisenberg(rho_Es,gammas,thetas,thetas,delta_ts,15,15,True,2000,25000,False,True,"Haar_42_47_short_direct")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(" Overall program runtime : ", time.time() - overall_start_time)
