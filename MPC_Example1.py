import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

# Initialization Value
x_initial, y_initial, theta_initial = 0, 0, 0
x_final, y_final = 5, 8
linear_velocity = 1.0
dt = 0.1
turn_radius = 1.0
cost_history = []

# Mobile Robot Model
def mobile_robot_model(state, angular_velocity):
    x, y, theta = state
    x_next = x + linear_velocity * np.cos(theta) * dt
    y_next = y + linear_velocity * np.sin(theta) * dt
    theta_next = theta + (linear_velocity * np.tan(angular_velocity) / turn_radius) * dt
    return np.array([x_next, y_next, theta_next])

# Update Trajectory
def update_positions_orientation(angular_velocities):
    state = np.array([x_initial, y_initial, theta_initial])
    positions = [state[:2]]
    for angular_velocity in angular_velocities:
        state = mobile_robot_model(state, angular_velocity)
        positions.append(state[:2])
    return positions

# Objective Function
def objective_function(angular_velocities):
    positions = update_positions_orientation(angular_velocities)
    final_position = positions[-1]
    final_position_cost = np.linalg.norm(final_position - [x_final, y_final])
    cost_history.append(final_position_cost)
    return final_position_cost

# MPC Control
horizon = 100
control_input = np.zeros(horizon)

initial_time = time.time()

# Solve the optimization problem
result = minimize(objective_function, control_input, method='SLSQP', options={'disp': True})

final_time = time.time()

print("Total Time = ", (final_time - initial_time))


# Result
optimal_angular_velocities = result.x
optimal_positions = update_positions_orientation(optimal_angular_velocities)

x_values = [pos[0] for pos in optimal_positions]
y_values = [pos[1] for pos in optimal_positions]

cost_history = np.array(cost_history)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
ax1.plot(x_values, y_values, linestyle='-', color='b')
ax1.scatter([x_initial], [y_initial], color='g', label='Start', s=50)
ax1.scatter([x_final], [y_final], color='r', label='End', s=50)

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Moblie Robot Path')
ax1.legend()
ax1.grid()
ax1.axis('equal')

ax2.plot(cost_history, marker='o', linestyle='-', color='blue')
ax2.set_title('Cost History over Iterations')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost')
ax2.grid(True)

plt.tight_layout()
plt.show()
