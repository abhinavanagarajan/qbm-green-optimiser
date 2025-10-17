import pennylane as qml
from pennylane import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Model configuration
n_visible = 8  # Input features: category, sequestration_rate, pollution_level, air_quality_index, plant_age, temperature, soil_moisture, light_intensity
n_hidden = 6   # Latent variables for capturing complex relationships
dev = qml.device('default.qubit', wires=n_visible + n_hidden)

def load_and_preprocess_data():
    """Load and preprocess the dataset for QBM training."""
    df = pd.read_csv('/Users/abhinavanagarajan/repos/GitHub/qbm-green-optimiser/data/dataset.csv')
    
    # Convert categorical data to numeric
    category_mapping = {'Deciduous': 0, 'Coniferous': 1, 'Grassland': 2}
    df['category_encoded'] = df['category'].map(category_mapping)
    
    # Select and normalize features
    features = [
        'category_encoded', 'sequestration_rate', 'pollution_level',
        'air_quality_index', 'plant_age', 'ambient_temperature',
        'soil_moisture', 'light_intensity'
    ]
    
    X = df[features].values
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Create target variables based on sustainability metrics
    y = np.column_stack([
        df['carbon_capture'] / df['carbon_capture'].max(),  # Normalized carbon capture
        df['sequestration_rate'] / df['sequestration_rate'].max(),  # Normalized sequestration rate
        df['air_quality_index'] / df['air_quality_index'].max(),  # Normalized air quality impact
        (1 - df['pollution_level'] / df['pollution_level'].max()),  # Inverse normalized pollution level
        df['soil_moisture'] / 100,  # Normalized soil health
        df['light_intensity'] / 1000  # Normalized light efficiency
    ])
    
    return X_normalized, y, scaler, category_mapping

@qml.qnode(dev)
def qbm_circuit(params, visible_data):
    """Enhanced QBM circuit with improved feature interactions."""
    # Encode input features
    for i in range(n_visible):
        qml.RY(visible_data[i] * np.pi, wires=i)
    
    # First layer of trainable rotations
    for i in range(n_visible + n_hidden):
        qml.RY(params[i], wires=i)
    
    # Entanglement layer between visible and hidden
    for i in range(n_visible):
        qml.CNOT(wires=[i, n_visible + (i % n_hidden)])
    
    # Second layer of trainable rotations for hidden units
    for i in range(n_visible, n_visible + n_hidden):
        qml.RY(params[i + n_visible + n_hidden], wires=i)
    
    # Additional entanglement between hidden units
    for i in range(n_hidden - 1):
        qml.CNOT(wires=[n_visible + i, n_visible + i + 1])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_visible, n_visible + n_hidden)]

def train_qbm(X_train, y_train, epochs=300, stepsize=0.05):
    """Train the QBM with enhanced optimization."""
    # Initialize parameters for both rotation layers
    params = np.random.randn(2 * (n_visible + n_hidden))
    optimizer = qml.AdamOptimizer(stepsize)
    
    best_params = None
    best_loss = float('inf')
    
    for epoch in range(epochs):
        def cost(params):
            predictions = np.array([qbm_circuit(params, x) for x in X_train])
            return np.mean((predictions - y_train) ** 2)
        
        params = optimizer.step(cost, params)
        current_loss = cost(params)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params.copy()
        
        if epoch % 20 == 0:
            print(f"[Epoch {epoch}] Loss: {current_loss:.6f}")
    
    return best_params

def analyze_sustainability(params, sample_input, scaler, category_mapping):
    """Analyze sustainability metrics and provide insights."""
    # Get QBM predictions
    predictions = qbm_circuit(params, sample_input)
    
    # Interpret the predictions
    carbon_efficiency = predictions[0]
    sequestration_potential = predictions[1]
    air_quality_impact = predictions[2]
    pollution_resistance = predictions[3]
    soil_health = predictions[4]
    light_efficiency = predictions[5]
    
    # Generate insights
    insights = {
        'carbon_efficiency': float(carbon_efficiency),
        'sequestration_potential': float(sequestration_potential),
        'air_quality_impact': float(air_quality_impact),
        'pollution_resistance': float(pollution_resistance),
        'soil_health': float(soil_health),
        'light_efficiency': float(light_efficiency)
    }
    
    # Generate recommendations
    recommendations = []
    if sequestration_potential < 0.4:
        recommendations.append("Consider increasing plant density for better carbon sequestration")
    if pollution_resistance < 0.5:
        recommendations.append("Implement additional pollution resistance measures")
    if soil_health < 0.6:
        recommendations.append("Soil health improvement recommended")
    if light_efficiency < 0.5:
        recommendations.append("Optimize plant placement for better light exposure")
    
    return insights, recommendations

def get_adaptation_score(insights):
    """Calculate overall adaptation score based on sustainability metrics."""
    weights = {
        'carbon_efficiency': 0.25,
        'sequestration_potential': 0.2,
        'air_quality_impact': 0.2,
        'pollution_resistance': 0.15,
        'soil_health': 0.1,
        'light_efficiency': 0.1
    }
    
    score = sum(insights[key] * weights[key] for key in weights)
    return score * 100  # Convert to percentage
