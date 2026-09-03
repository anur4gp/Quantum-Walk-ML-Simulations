#main

"""Pre-restructure script, kept verbatim apart from the import shim below.

Superseded by scripts/plot_channels.py, which drives the ported physics in
src/qw/ from a config file.
"""

import sys
from pathlib import Path

# Kept runnable after the restructure: walker.py now lives in src/qw/legacy/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "qw" / "legacy"))

from walker import Walker
import numpy as np
import matplotlib.pyplot as plt

# Task: Extract the spin-up and spin-down probabilities from the walkMap and plot them against the positions.
# Then, plot the overall probability distribution P(x) for the Hadamard coin.

steps = 300
theta_base = np.pi/4
phi1 = np.pi/2
phi2 = np.pi/2
delta_theta = np.pi/8
pR = 0.5

# 1. Create the walker --300 to 300 lattice
w = Walker(steps)

# 2. Run a standard quantum walk w/ Hadamard coin (theta = pi/4, phi1 = pi/2, phi2 = pi/2)
w.runWalk(theta_base, phi1, phi2)

# Random walker with jittered coin angles (theta_base = pi/4, delta_theta = pi/8, pR = 0.5)
w_DRC = Walker.runDRCoinWalk(300, theta_base, phi1, phi2, delta_theta, pR)

# Random walker with uniformly sampled jittered coin angles (theta_base = pi/4, delta_theta = pi/8, pR = 0.5)
w_CRC = Walker.runCRCoinWalk(300, theta_base, phi1, phi2, delta_theta)

W_RT = Walker.runRTransWalk(300, theta_base, phi1, phi2, 0.5)

fig, ax = plt.subplots()
ax.plot(w.positions, w.probabilites, 
        label="Pure QW", color="blue")
ax.plot(w_DRC.positions, w_DRC.probabilites, 
        label=f"Jittered", color="red", linestyle="dashed")
ax.plot(w_CRC.positions, w_CRC.probabilites,
        label=f"Uniform Jittered", color="green", linestyle="dotted")
ax.plot(W_RT.positions, W_RT.probabilites,
        label=f"Random Transform", color="purple", linestyle="dashdot")
ax.set_xlabel("x")
ax.set_ylabel("P(x)")
ax.set_title(f"Pure vs. Random Walk (Discreet Coin, Continuous Coin, Continuous Translate), {w.maxSteps} Steps")
ax.legend()
plt.show()