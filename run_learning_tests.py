import numpy as np
import matplotlib.pyplot as plt

print("===============================================================")
print("      ODU DIGITAL TWIN MULTI-STAGE LEARNING TEST SUITE         ")
print("===============================================================")

# Generate synthetic flight telemetry logs
np.random.seed(42)
flight_hours = np.linspace(0, 10, 100)
true_fatigue = 0.02 * (flight_hours ** 1.5) + np.random.normal(0, 0.01, 100)

# Define 3 separate test scenarios showing improvement via learning configurations
scenarios = [
    {"name": "Test Run 1: High Error Baseline", "epochs": 50, "lr": 1e-4},
    {"name": "Test Run 2: Extended Learning Optimization", "epochs": 200, "lr": 5e-4},
    {"name": "Test Run 3: Fully Converged Production Twin", "epochs": 500, "lr": 1e-3}
]

plt.figure(figsize=(10, 5))

# Execute successive training loops to showcase learning tracking history
for run_idx, config in enumerate(scenarios, start=1):
    print(f"\n🚀 Executing {config['name']}...")
    weight = 0.0
    loss_history = []
    
    for epoch in range(config["epochs"]):
        predictions = weight * (flight_hours ** 1.5)
        error = predictions - true_fatigue
        gradient = (2 / len(flight_hours)) * np.sum(error * (flight_hours ** 1.5))
        weight -= config["lr"] * gradient
        
        # Track historical loss calculation indices
        current_loss = np.mean(error ** 2)
        loss_history.append(current_loss)
        
    final_loss = loss_history[-1]
    print(f"--> Status: Complete | Final Tracking Loss Index: {final_loss:.6f} | Factor: {weight:.4f}")
    
    # Plot performance histories to visually demonstrate improvement curves
    plt.plot(loss_history, label=f"{config['name']} (Final Loss: {final_loss:.5f})", linewidth=2)

# Configure professional data science plot properties
plt.title("Digital Twin Multilateral Optimization: Historical Loss Convergence", fontsize=12, fontweight='bold')
plt.xlabel("Training Epoch (Learning Step Progression)", fontsize=10)
plt.ylabel("Mean Squared Error (System Accuracy Index)", fontsize=10)
plt.yscale('log') # Log scale to clearly highlight refinement bounds
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()

# Save visual metrics to showcase results to faculty or employers
output_chart = "twin_learning_convergence_matrix.png"
plt.savefig(output_chart, dpi=300, bbox_inches='tight')
plt.close()

print("\n===============================================================")
print(f"SUCCESS: Learning analysis chart exported as '{output_chart}'!")
print("===============================================================")
