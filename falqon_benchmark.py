import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import time

edges = [(0, 1), (1, 2), (2, 0), (2, 3), (1, 4)]
graph = nx.Graph(edges)

pos = nx.spring_layout(graph)
nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='black')
plt.show()

# Define cost and driver Hamiltonians for Vertex Cover
cost_h, driver_h = qml.qaoa.min_vertex_cover(graph, constrained=False)

# Function to build commutator Hamiltonian for Vertex Cover
def build_commutator_hamiltonian(graph):
    H = qml.Hamiltonian([], [])
    graph_complement = nx.complement(graph)
    for node in graph_complement.nodes:
        for edge in graph_complement.edges:
            i, j = edge
            if node == i:
                H += 6 * (qml.PauliY(node) @ qml.PauliZ(j) - qml.PauliY(node))
            if node == j:
                H += 6 * (qml.PauliZ(i) @ qml.PauliY(node) - qml.PauliY(node))
        H += 6 * qml.PauliY(node)
    return H

def falqon_layer(beta_k, cost_h, driver_h, delta_t):
    qml.ApproxTimeEvolution(cost_h, delta_t, 1)
    qml.ApproxTimeEvolution(driver_h, delta_t * beta_k, 1)

# Define FALQON ansatz
def create_vertex_cover_ansatz(cost_h, driver_h, delta_t):
    def ansatz(beta, **kwargs):
        for wire in dev.wires:
            qml.Hadamard(wires=wire)
        qml.layer(falqon_layer, len(beta), beta, cost_h=cost_h, driver_h=driver_h, delta_t=delta_t)
    return ansatz

def expectation_value_circuit(beta, measurement_h):
    ansatz = create_vertex_cover_ansatz(cost_h, driver_h, delta_t)
    ansatz(beta)
    return qml.expval(measurement_h)

# Implement FALQON algorithm
def run_vertex_cover_falqon(graph, iterations, initial_beta, delta_t, device):
    commutator_h = build_commutator_hamiltonian(graph)
    cost_h, driver_h = qml.qaoa.min_vertex_cover(graph, constrained=False)
    cost_function = qml.QNode(expectation_value_circuit, device, interface="autograd")
    beta_values = [initial_beta]
    energy_values = []
    start_time = time.time()
    for _ in range(iterations):
        beta_values.append(-1 * cost_function(beta_values, measurement_h=commutator_h))
        energy = cost_function(beta_values, measurement_h=cost_h)
        energy_values.append(energy)
    exec_time = time.time() - start_time
    return beta_values, energy_values, exec_time

# Set parameters
iterations = 40
initial_beta = 0.0
delta_t = 0.01

# Create quantum device
dev = qml.device("default.qubit", wires=graph.nodes)


resulting_beta, resulting_energies, execution_time = run_vertex_cover_falqon(graph, iterations, initial_beta, delta_t, dev)

mean_energy_falqon = np.mean(resulting_energies)
variance_energy_falqon = np.var(resulting_energies)
convergence_rate_falqon = (resulting_energies[0] - resulting_energies[-1]) / len(resulting_energies)

# Create probability circuit
@qml.qnode(dev, interface="autograd")
def probability_circuit():
    ansatz = create_vertex_cover_ansatz(cost_h, driver_h, delta_t)
    ansatz(resulting_beta)
    return qml.probs(wires=dev.wires)

# Plot probability distribution
probabilities = probability_circuit()
plt.figure(figsize=(10, 6))
plt.bar(range(2**len(dev.wires)), probabilities)
plt.xlabel("Bit string")
plt.ylabel("Measurement Probability")
plt.title("Probability Distribution After FALQON")
plt.show()

new_edges = [(0, 1), (1, 2), (2, 0), (2, 3), (1, 4), (4, 5), (5, 2), (0, 6)]
new_graph = nx.Graph(new_edges)

depth = 5
dev = qml.device("default.qubit", wires=new_graph.nodes)
cost_h, mixer_h = qml.qaoa.min_vertex_cover(new_graph, constrained=False)

def qaoa_layer(gamma, beta):
    qml.qaoa.cost_layer(gamma, cost_h)
    qml.qaoa.mixer_layer(beta, mixer_h)

def qaoa_circuit(params, **kwargs):
    for wire in dev.wires:
        qml.Hadamard(wires=wire)
    qml.layer(qaoa_layer, depth, params[0], params[1])

@qml.qnode(dev, interface="autograd")
def qaoa_expectation_value(params):
    qaoa_circuit(params)
    return qml.expval(cost_h)

# Run FALQON for initial QAOA parameters
delta_t = 0.05
res, res_energy, falqon_exec_time = run_vertex_cover_falqon(new_graph, depth-1, 0.0, delta_t, dev)
params = np.array([[delta_t for k in res], [delta_t * k for k in res]], requires_grad=True)

# Optimize QAOA parameters
steps = 40
optimizer = qml.GradientDescentOptimizer()
qaoa_energies = []
start_time = time.time()
for step in range(steps):
    params, cost = optimizer.step_and_cost(qaoa_expectation_value, params)
    qaoa_energies.append(cost)
qaoa_exec_time = time.time() - start_time

# Define probability circuit for optimized QAOA
@qml.qnode(dev, interface="autograd")
def probability_circuit_qaoa(params):
    qaoa_circuit(params)
    return qml.probs(wires=dev.wires)

# Plot probability distribution for optimized QAOA
probabilities_qaoa = probability_circuit_qaoa(params)
plt.figure(figsize=(10, 6))
plt.bar(range(2**len(dev.wires)), probabilities_qaoa)
plt.xlabel("Bit string")
plt.ylabel("Measurement Probability")
plt.title("Probability Distribution After QAOA")
plt.show()

# Ensure bit string matches expected values
bitstrings = [bin(i)[2:].zfill(len(new_graph.nodes)) for i in range(2**len(new_graph.nodes))]
most_probable_index = np.argmax(probabilities_qaoa)
most_probable_bitstring = bitstrings[most_probable_index]
print(f"Most probable bit string: {most_probable_bitstring}")

mean_energy_qaoa = np.mean(qaoa_energies)
variance_energy_qaoa = np.var(qaoa_energies)
convergence_rate_qaoa = (qaoa_energies[0] - qaoa_energies[-1]) / len(qaoa_energies)

# Compare FALQON and QAOA
plt.figure(figsize=(10, 6))
plt.plot(range(1, iterations + 1), resulting_energies, label='FALQON Energy')
plt.plot(range(steps), qaoa_energies, label='QAOA Energy', linestyle='--')
plt.xlabel("Iteration")
plt.ylabel("Cost Function Value")
plt.title("FALQON vs QAOA Energy Over Iterations")
plt.legend()
plt.grid()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0, 0].bar(['FALQON', 'QAOA'], [execution_time, qaoa_exec_time], color=['blue', 'orange'])
axs[0, 0].set_xlabel("Algorithm")
axs[0, 0].set_ylabel("Execution Time (seconds)")
axs[0, 0].set_title("Execution Time Comparison")

axs[0, 1].bar(['FALQON', 'QAOA'], [mean_energy_falqon, mean_energy_qaoa], color=['blue', 'orange'])
axs[0, 1].set_xlabel("Algorithm")
axs[0, 1].set_ylabel("Mean Energy")
axs[0, 1].set_title("Mean Energy Comparison")

axs[1, 0].bar(['FALQON', 'QAOA'], [variance_energy_falqon, variance_energy_qaoa], color=['blue', 'orange'])
axs[1, 0].set_xlabel("Algorithm")
axs[1, 0].set_ylabel("Energy Variance")
axs[1, 0].set_title("Energy Variance Comparison")

axs[1, 1].bar(['FALQON', 'QAOA'], [convergence_rate_falqon, convergence_rate_qaoa], color=['blue', 'orange'])
axs[1, 1].set_xlabel("Algorithm")
axs[1, 1].set_ylabel("Convergence Rate")
axs[1, 1].set_title("Convergence Rate Comparison")

plt.tight_layout()
plt.show()

print("\nFALQON Statistics:")
print(f"Mean Energy: {mean_energy_falqon}")
print(f"Energy Variance: {variance_energy_falqon}")
print(f"Convergence Rate: {convergence_rate_falqon}")

print("\nQAOA Statistics:")
print(f"Mean Energy: {mean_energy_qaoa}")
print(f"Energy Variance: {variance_energy_qaoa}")
print(f"Convergence Rate: {convergence_rate_qaoa}")
