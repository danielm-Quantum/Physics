# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 23:14:51 2025

@author: daniel
"""

import numpy as np
from qutip import coherent, num, expect

# Define the coherent state parameter alpha = x + i y
x = 0.5* np.pi  # Real part set to pi
y = np.log(2)  /30  # Imaginary part 
alpha = x + 1j * y  # Complex alpha


N = 100
coherent_state = coherent(N, alpha)
number_operator = num(N)
mean_n = expect(number_operator, coherent_state)

mean_n = expect(num(N), coherent(N, alpha))

print(f"For alpha = {alpha} , the mean photon number <n> = {mean_n:.6f}")
print(f"Analytical check: |alpha|^2 = {abs(alpha)**2:.6f}")
