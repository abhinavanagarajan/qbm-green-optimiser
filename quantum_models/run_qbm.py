import numpy as np
from qbm_model import (
    load_and_preprocess_data,
    train_qbm,
    analyze_sustainability,
    get_adaptation_score
)

def print_plant_insights(insights, recommendations, adaptation_score):
    """Print detailed sustainability insights and recommendations."""
    print("\n=== Plant Sustainability Analysis ===")
    print(f"\nOverall Adaptation Score: {adaptation_score:.1f}%\n")
    
    print("📊 Sustainability Metrics:")
    metrics = {
        'carbon_efficiency': ('Carbon Capture Efficiency', '🌳'),
        'sequestration_potential': ('CO2 Sequestration Potential', '💨'),
        'air_quality_impact': ('Air Quality Impact', '🌬️'),
        'pollution_resistance': ('Pollution Resistance', '🛡️'),
        'soil_health': ('Soil Health', '🌱'),
        'light_efficiency': ('Light Utilization', '☀️')
    }
    
    for key, (label, emoji) in metrics.items():
        value = insights[key]
        bars = '█' * int(value * 20)
        spaces = '░' * (20 - int(value * 20))
        print(f"{emoji} {label}: {bars}{spaces} {value*100:.1f}%")
    
    if recommendations:
        print("\n🔍 Recommendations for Improvement:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

def main():
    print("Loading and preprocessing data...")
    X_train, y_train, scaler, category_mapping = load_and_preprocess_data()
    
    print("\n🧠 Training Quantum Boltzmann Machine...")
    trained_params = train_qbm(X_train, y_train)
    
    # Generate sample input (you can modify these values)
    sample_input = X_train[0]  # Using first data point as example
    
    print("\n📈 Analyzing plant sustainability...")
    insights, recommendations = analyze_sustainability(trained_params, sample_input, scaler, category_mapping)
    adaptation_score = get_adaptation_score(insights)
    
    print_plant_insights(insights, recommendations, adaptation_score)

if __name__ == "__main__":
    main()
