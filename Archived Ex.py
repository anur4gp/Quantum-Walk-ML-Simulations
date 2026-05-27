# Archived exercises

from walker import Walker
import numpy as np
import matplotlib.pyplot as plt

steps = 300
theta_base = np.pi/4
phi1 = np.pi/2
phi2 = np.pi/2
delta_theta = np.pi/8
pR = 0.5

w = Walker(steps)
w.runWalk(theta_base, phi1, phi2)

positions = w.positions

p_plus  = np.abs(w.walkMap[:, 0])**2
p_minus = np.abs(w.walkMap[:, 1])**2


fig, ax = plt.subplots()
ax.plot(positions, p_plus,  label="$P_+$", color="blue")
ax.plot(positions, p_minus, label="$P_-$", color="red", linestyle="dashed")
ax.set_xlabel("x")
ax.set_ylabel("Probability P(x)")
ax.set_title(f"Spin-up vs Spin-down, {w.maxSteps} Steps")
ax.legend()
plt.show()


w.probDistribution("Hadamard")
plt.show()


thetas = np.linspace(-np.pi/2, np.pi/2, steps)  # sweep theta from -pi/2 to pi/2 using 'steps' points

delta_x = [] # Array used to store the distance of the P+ peak from the starting position (x=0) for each theta

for theta in thetas: #
    w = Walker(steps)
    w.runWalk(theta, np.pi/4, np.pi/4)

    p_plus = np.abs(w.walkMap[:, 0])**2

    # Index of the P+ peak
    peak_index = np.argmax(p_plus)

    # Convert index to actual x position (distance from start at x=0)
    delta_x_m = abs(w.positions[peak_index])
    delta_x.append(delta_x_m)

# Plotting position vs. coin angle (theta)
fig, ax = plt.subplots()
ax.plot(thetas, delta_x, color="blue")
ax.set_xlabel(r"$\theta$ (radians)")
ax.set_ylabel(r"$\Delta x_m$")
ax.set_title(f"Peak Spin-up Position vs Coin Angle, {steps} Steps")
ax.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
ax.set_xticklabels(["-$\pi/2$", r"-$\pi/4$", r"0", r"$\pi/4$", r"$\pi$/2"])
plt.show()