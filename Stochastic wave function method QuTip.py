# -*- coding: utf-8 -*-
"""
Created on Oct 2025

@author: daniel
"""
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

# Function to define  operators for dimension N (all are Qobj)
def define_operators(N):
    
    a = qt.destroy(N)          # Annihilation operator
    a_dag = qt.create(N)       # Creation operator     (Hermitian conjugate of a)
    n = qt.num(N)              # Number operator        n = a_dag * a
    x = qt.position(N)         # Position operator      x = (a + a_dag) / sqrt(2)
    p = qt.momentum(N)         # Momentum operator      p = -1j * (a - a_dag) / sqrt(2)
 
    return a, a_dag, n, x, p


# User input: Select Hilbert space dimension N
N = int(input("Enter the Hilbert space dimension N (e.g., 10 for harmonic oscillator cutoff): "))

# Define operators based on N
a, a_dag, n, x, p = define_operators(N)

# Define Hamiltonian H 

omega = np.pi  # Frequency
zeta = np.pi**2 + (np.log(2)/10)**2
H = omega * n + zeta*(a_dag+a)

# H = omega * n  


# Define Lindblad operators Ls 
# For a harmonic oscillator with damping: Ls = [sqrt(gamma) * a]

k = 2* np.log(2)/10  # Damping rate
Ls = [np.sqrt(k) * a]  # Single Lindblad for photon loss

# Alternative: Add more, e.g., Ls = [np.sqrt(gamma) * a, np.sqrt(gamma/2) * a_dag]

# Compute effective Hamiltonian Heff

Heff = H - 1j/2 * sum([L.dag() * L for L in Ls])

# Define initial state psi0 
# For harmonic oscillator: Coherent state or vacuum
# Vacuum state |0>
psi0 = qt.basis(N, 0)
# Alternative: First excited state |1> (as in original code's psi0[1] = 1.0)
# psi0 = qt.basis(N, 1)
# Alternative: Coherent state (approximate for small alpha)
# alpha = 1.0
# psi0 = qt.coherent(N, alpha)
# For simplicity, using vacuum; replace with your state.

# Timestep and simulation parameters
dt = 0.9 / H.norm(norm='fro')
m = 300  # Number of steps
tf = dt * m  # Final time
times = np.linspace(0, tf, m)



# Monte Carlo simulation
sample = 100  # Number of samples
mean = np.zeros(m, dtype=complex)  # Array for results (e.g., mean of some observable)


for count in range(sample):
    t = 0
    waves = [psi0]
    for t in times[1:]:
        # Generate a random number in (0,1]
        u = np.random.random()
        # Array of jump probabilities
        # Adjusted for Qobj: use .dag() and * for Hermitian conjugate and multiplication
        dps = [np.real(dt * (waves[-1].dag() * (L.dag() * L) * waves[-1])) for L in Ls]
        # Renormalization factor dP
        dP = np.sum(dps)
        # Test for jump
        if dP < u:
            temp = (qt.qeye(N) - 1j * Heff.dag() * dt) * waves[-1]
        else:
            # New random number
            u = np.random.random()
            Q = np.cumsum(dps) / dP
            # Pick the jump that occurred
            k = np.searchsorted(Q, u, side='left')
            temp = Ls[k] * waves[-1]
        # Normalize and append
        waves.append(temp / temp.norm())

    # Example: Compute mean of <n> (number operator expectation) over time
    # You can change this to any observable, e.g., <x> or <p>
    # Adjusted: use Qobj expectation value method
    n_expect = [qt.expect(n, waves[i]) for i in range(m)]
    mean += np.array(n_expect) / sample  # Average over samples

# Plot the results (example: mean <n> over time)
plt.figure(figsize=(8, 5))
plt.plot(times, np.real(mean), label=r'$\langle n \rangle$')
plt.xlabel('Time')
plt.ylabel(r'$\langle n \rangle$')
plt.title(f'Time Evolution of Mean Number Operator (Stochastic QuTiP) (N={N}) (dt={dt}')
plt.legend()
plt.grid(True)
plt.show()


