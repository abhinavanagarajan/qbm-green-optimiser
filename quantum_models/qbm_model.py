import pennylane as qml
from pennylane import numpy as np

n_visible = 6
n_hidden = 4
dev = qml.device('default.qubit', wires=n_visible + n_hidden)

@qml.qnode(dev)
def qbm_circuit(params, visible_data):
    for i in range(n_visible):
        qml.RY(visible_data[i], wires=i)
    for i in range(n_visible + n_hidden):
        qml.RY(params[i], wires=i)
    for i in range(n_visible):
        qml.CNOT(wires=[i, n_visible + (i % n_hidden)])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_hidden)]

def train_qbm(X_train, y_train, epochs=200, stepsize=0.1):
    params = np.random.randn(n_visible + n_hidden)
    optimizer = qml.NesterovMomentumOptimizer(stepsize)

    for epoch in range(epochs):
        def cost(params):
            predictions = np.array([qbm_circuit(params, x) for x in X_train])
            return np.mean((predictions - y_train) ** 2)

        params = optimizer.step(cost, params)

        if epoch % 20 == 0:
            print(f"[Epoch {epoch}] Loss: {cost(params):.6f}")

    return params

def sample_qbm(params, sample_input):
    return qbm_circuit(params, sample_input)
