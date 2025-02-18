import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(0, x)+alpha*np.minimum(0, x)

# Generate input values
x = np.linspace(-10, 10, 400)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)

# Create figure with subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the ReLU function
axs[0].plot(x, y_relu, label='ReLU(x)', color='b', linewidth=2)
axs[0].axhline(0, color='black', linewidth=0.5, linestyle='--')
axs[0].axvline(0, color='black', linewidth=0.5, linestyle='--')
axs[0].set_title("ReLU Activation Function")
axs[0].set_xlabel("x")
axs[0].set_ylabel("ReLU(x)")
axs[0].legend()
axs[0].grid(True)

# Plot the Leaky ReLU function
axs[1].plot(x, y_leaky_relu, label='Leaky ReLU(x)', color='r', linewidth=2)
axs[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
axs[1].axvline(0, color='black', linewidth=0.5, linestyle='--')
axs[1].set_title("Leaky ReLU Activation Function")
axs[1].set_xlabel("x")
axs[1].set_ylabel("Leaky ReLU(x)")
axs[1].legend()
axs[1].grid(True)
plt.savefig(r'Machine Learning\Plots\relu.png',dpi=200)
plt.show()


