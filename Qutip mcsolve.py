# -*- coding: utf-8 -*-
"""
Created on Oct 2025

@author: daniel
"""
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# User input: Select Hilbert space dimension N
N = int(input("Enter the Hilbert space dimension N (e.g., 10 for harmonic oscillator cutoff): "))

# Define operators using QuTiP (all are Qobj)
a = qt.destroy(N)          # Annihilation operator
a_dag = qt.create(N)       # Creation operator
n = qt.num(N)              # Number operator n = a_dag * a
x = qt.position(N)         # Position operator
p = qt.momentum(N)         # Momentum operator

# Define Hamiltonian H 

omega = np.pi  # Frequency
zeta = np.pi**2 + (np.log(2)/10)**2
H = omega * n + zeta*(a_dag+a)
# H = omega * (n + 0.5 * qt.qeye(N))  # Qobj Hamiltonian (qt.qye(N)= identity of N dim)
# Alternative: H = omega * n


# Define Lindblad operators Ls (list of Qobj)
# For photon loss: Ls = [sqrt(gamma) * a]
k = 2* np.log(2)/10  # Damping rate
Ls = [np.sqrt(k) * a]  # Single Lindblad for loss
# Alternative: Add more, e.g., Ls = [np.sqrt(gamma) * a, np.sqrt(gamma/2) * a_dag]




# Define initial state psi0 (Qobj, customize as needed)
# Vacuum state |0>
psi0 = qt.basis(N, 0)

# Alternative: Coherent state |alpha>
# alpha = 1.0 
# alpha= 0.5* np.pi +  1j*(np.log(2)/30)
# psi0 = qt.coherent(N, alpha)

# Time list for simulation
tlist = np.linspace(0, 50, 201)  # From t=0 to t=20, 201 points (adjust as needed)

# Monte Carlo simulation using QuTiP's mcsolve
# ntraj: Number of trajectories (samples)
# e_ops: Expectation operators to compute (e.g., <n>)
result = qt.mcsolve(H, psi0, tlist, Ls, e_ops=[n], ntraj=1000, options=qt.Options(store_states=False))
# - H: Hamiltonian (Qobj)
# - psi0: Initial state (Qobj)
# - tlist: Time points
# - Ls: List of Lindblad operators (Qobj)
# - e_ops: List of observables to average (e.g., [n] for <n>)
# - ntraj: Number of Monte Carlo trajectories (samples)
# - options: store_states=False to save memory (only compute expectations)

# Extract results: result.expect[0] is the mean <n> over trajectories and time
mean_n = result.expect[0]  # Array of <n> at each time in tlist

target_time = 10.0
index = np.argmin(np.abs(tlist - target_time))  # Closest index to t=10
print(f"At time t = {tlist[index]:.2f}, the mean number operator <n> = {mean_n[index]:.6f}")


# Plot the results
plt.figure(figsize=(8, 5))
plt.plot(tlist, mean_n, label=r'$\langle n \rangle$')
plt.xlabel('Time')
plt.ylabel(r'$\langle n \rangle$')
plt.title('Time Evolution of Mean Number Operator (QuTiP Monte Carlo)')
plt.title(f'Time Evolution of Mean Number Operator (QuTiP Monte Carlo) (N={N})')
plt.legend()
plt.grid(True)
plt.show()


# Optional: Access individual trajectories if needed (set store_states=True in options)
# For example, to inspect a specific trajectory:
# result.states[0]  # List of states for trajectory 0 (if store_states=True)