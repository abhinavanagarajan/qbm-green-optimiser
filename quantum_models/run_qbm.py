import numpy as np
from qbm_model import train_qbm, sample_qbm

# Simulated Dataset
X_train = np.random.uniform(low=0.0, high=1.0, size=(500, 6))   # Pollution + Greenery + Geo Features
y_train = np.random.uniform(low=0.0, high=1.0, size=(500, 4))  # Latent variables

print("🔧 Training Quantum Boltzmann Machine...")
trained_params = train_qbm(X_train, y_train, epochs=200, stepsize=0.1)

print("\n🎯 Sampling New Synthetic Pollution Scenario...")
sample_input = np.random.uniform(size=6)
sampled_output = sample_qbm(trained_params, sample_input)
print("📊 Sampled Latent Variables:", sampled_output)
