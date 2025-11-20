# -*- coding: utf-8 -*-
"""
Created on Wed Oct 2025

@author: daniel
"""

import numpy as np
import matplotlib.pyplot as plt

# Function to define creation and annihilation operators for dimension N
def define_operators(N):
    # Annihilation operator a
    a = np.zeros((N, N), dtype=complex)
    for i in range(N-1):
        a[i, i+1] = np.sqrt(i+1)
    
    # Creation operator a_dag (Hermitian conjugate of a)
    a_dag = a.T.conjugate()
    
    # Number operator n = a_dag @ a
    n = a_dag @ a
    
    # Position operator x = (a + a_dag) / sqrt(2)
    x = (a + a_dag) / np.sqrt(2)
    
    # Momentum operator p = -1j * (a - a_dag) / sqrt(2)
    p = -1j * (a - a_dag) / np.sqrt(2)
    
    return a, a_dag, n, x, p

# User input: Select Hilbert space dimension N
N = int(input("Enter the Hilbert space dimension N "))

# Define operators based on N
a, a_dag, n, x, p = define_operators(N)

# Define Hamiltonian H 
# For a harmonic oscillator: H = omega * (n + 0.5) + some perturbation

omega = np.pi  # Frequency
zeta = np.pi**2 + (np.log(2)/10)**2
H = omega * n + zeta*(a_dag+a)



# omega = 1.0  # Frequency
# H = omega * (n + 0.5 * np.eye(N))  # Example: Harmonic oscillator Hamiltonian
# Alternative: H = omega * n  # Simplified version
# You can modify H using a, a_dag, n, x, p, e.g., H = omega * n + g * (a + a_dag)

# Define Lindblad operators Ls 
# For a harmonic oscillator with damping: Ls = [sqrt(gamma) * a]
gamma =  2* np.log(2)/10 # Damping rate
Ls = [np.sqrt(gamma) * a]  # Example: Single Lindblad operator for photon loss
# Alternative: Add more, e.g., Ls = [np.sqrt(gamma) * a, np.sqrt(gamma/2) * a_dag]

# Compute effective Hamiltonian Heff
Heff = H - 1j/2 * sum([L.T.conjugate() @ L for L in Ls])

# Define initial state psi0 
# For harmonic oscillator: Coherent state or vacuum
# Vacuum state |0>
psi0 = np.zeros(N, dtype=complex)
psi0[0] = 1.0
# Alternative: Coherent state (approximate for small alpha)
# alpha = 1.0
# psi0 = np.exp(-0.5 * abs(alpha)**2) * sum([alpha**k / np.sqrt(np.math.factorial(k)) * np.eye(N)[k, :] for k in range(N)])
# For simplicity, using vacuum; replace with your state.

# Timestep and simulation parameters
dt = 0.9 / np.linalg.norm(H, ord='fro')  # Adaptive timestep
m = 9000  # Number of steps
tf = dt * m  # Final time
times = np.linspace(0, tf, m)

# Monte Carlo simulation
sample = 50  # Number of samples
mean = np.zeros(m, dtype=complex)  # Array for results (e.g., mean of some observable)

for count in range(sample):
    t = 0
    waves = [psi0]
    for t in times[1:]:
        # Generate a random number in (0,1]
        u = np.random.random()
        # Array of jump probabilities
        dps = [np.real(dt * (waves[-1].T.conjugate() @ (L.T.conjugate() @ L) @ waves[-1])) for L in Ls]
        # Renormalization factor dP
        dP = np.sum(dps)
        # Test for jump
        if dP < u:
            temp = (np.eye(N) - 1j * Heff.T.conjugate() * dt) @ waves[-1]
        else:
            # New random number
            u = np.random.random()
            Q = np.cumsum(dps) / dP
            # Pick the jump that occurred
            k = np.searchsorted(Q, u, side='left')
            temp = Ls[k] @ waves[-1]
        # Normalize and append
        waves.append(temp / np.linalg.norm(temp))

    # Example: Compute mean of <n> (number operator expectation) over time
    # You can change this to any observable, e.g., <x> or <p>
    n_expect = [waves[i].T.conjugate() @ n @ waves[i] for i in range(m)]
    mean += np.array(n_expect) / sample  # Average over samples

# Plot the results (example: mean <n> over time)
plt.figure(figsize=(8, 5))
plt.plot(times, np.real(mean), label=r'$\langle n \rangle$')
plt.xlabel('Time')
plt.ylabel(r'$\langle n \rangle$')
plt.title(f'Time Evolution of Mean Number Operator (Stochastic wavefunction) (N={N}, dt={dt:.4e})')
plt.legend()
plt.grid(True)
plt.show()