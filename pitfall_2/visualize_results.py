import matplotlib.pyplot as plt

# Step 1: Prepare the data
leak_percentages = [0, 20, 40, 60, 80, 100]

data = {
    "Devign":     [0.616, 0.708, 0.780, 0.845, 0.913, 0.973],
    "DiverseVul": [0.6, 0.671, 0.728, 0.785, 0.839, 0.819],  # Placeholder
    "PrimeVul":   [0.617, 0.683, 0.749, 0.789, 0.855, 0.890],  # Placeholder
}

colors = {
    "Devign": "blue",
    "DiverseVul": "orange",
    "PrimeVul": "green"
}

# Step 2: Plot the data
plt.figure(figsize=(8, 6))
for dataset, f1_scores in data.items():
    plt.plot(leak_percentages, f1_scores, marker='o', label=dataset, color=colors[dataset])

plt.xlabel("Leakage Percentage")
plt.ylabel("F1 Score (Full Test)")
plt.title("F1 Score vs Leakage Percentage")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Step 3: Save the plot as a high-res PDF
plt.savefig("pitfall_2/generated_figures/f1_vs_leakage.pdf", format="pdf", dpi=300)