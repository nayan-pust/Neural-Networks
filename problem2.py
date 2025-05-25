import numpy as np
import matplotlib.pyplot as plt

# XOR input and expected output
inputs = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

targets = np.array([0, 1, 1, 0])

# Define McCulloch-Pitts neuron as a step function
def mcp_neuron(weight, threshold, x):
    return int(np.dot(weight, x) >= threshold)

# XOR using 2-layer MCP network
def xor_mcp(x1, x2):
    n1 = mcp_neuron([1, -1], 1, [x1, x2])   # x1 AND NOT x2
    n2 = mcp_neuron([-1, 1], 1, [x1, x2])   # NOT x1 AND x2
    y = mcp_neuron([1, 1], 1, [n1, n2])     # n1 OR n2
    return y

# Run XOR
outputs = np.array([xor_mcp(x[0], x[1]) for x in inputs])

# Display results
print("Input  =>  Output")
for i in range(4):
    print(f"{inputs[i]}  =>   {outputs[i]}")

# Accuracy
accuracy = np.mean(outputs == targets)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Convergence Curve (trivial: all correct in 1 step)
plt.figure()
plt.plot([1, 2, 3, 4], outputs == targets, 'bo-', label='Correct=1, Wrong=0')
plt.title('Convergence (trivial for MCP)')
plt.xlabel('Input Sample')
plt.ylabel('Correctness')
plt.ylim(-0.1, 1.1)
plt.grid()
plt.legend()
plt.show()

# Decision boundary plot
x = np.linspace(-0.2, 1.2, 200)
y = np.linspace(-0.2, 1.2, 200)
X, Y = np.meshgrid(x, y)
Z = np.array([[xor_mcp(xi, yi) for xi in x] for yi in y])

plt.figure(figsize=(7,7))
plt.contourf(X, Y, Z, levels=[-1, 0.5, 1], colors=['#1f77b4', '#ff7f0e'], alpha=0.7)  # nice blue and orange
scatter = plt.scatter(inputs[:,0], inputs[:,1], c=targets, cmap='bwr', edgecolors='black', s=150, marker='D', linewidth=1.5)
plt.title('XOR Decision Boundary (MCP Logic)', fontsize=16, fontweight='bold', color='#333333')
plt.xlabel('Input 1', fontsize=14, color='#555555')
plt.ylabel('Input 2', fontsize=14, color='#555555')
plt.grid(visible=True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.colorbar(scatter, ticks=[0,1], label='Target Class')
plt.show()
