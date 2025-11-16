"""
Generate performance visualization chart
"""
import matplotlib.pyplot as plt
import numpy as np

# Benchmark data
algorithms = [
    'Matrix Multiply', 'Statistics', 'Is Prime', 'Monte Carlo PI',
    'Merge Sort', 'Simpson Integration', 'Fibonacci', 'Primes Sieve',
    'Levenshtein', 'Factorial'
]
speedups = [458.57, 88.78, 56.91, 34.18, 30.66, 26.63, 21.14, 20.23, 15.40, 15.35]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Cython vs Python Performance Comparison', fontsize=16, fontweight='bold')

# Bar chart
colors = ['#e74c3c' if x > 100 else '#e67e22' if x > 50 else '#3498db' if x > 25 else '#2ecc71' for x in speedups]
ax1.barh(algorithms, speedups, color=colors, edgecolor='black', linewidth=0.5)
ax1.axvline(x=1, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='No speedup')
ax1.set_xlabel('Speedup Factor (x)', fontsize=12, fontweight='bold')
ax1.set_title('Performance Speedup by Algorithm', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.set_xlim(0, max(speedups) * 1.1)

# Add value labels
for i, v in enumerate(speedups):
    ax1.text(v + 10, i, f'{v:.1f}x', va='center', fontsize=10, fontweight='bold')

# Summary statistics
summary_data = {
    'Average': 76.78,
    'Median': 28.64,
    'Max': 458.57,
    'Min': 15.35
}

colors_summary = ['#3498db', '#2ecc71', '#e74c3c', '#95a5a6']
ax2.bar(summary_data.keys(), summary_data.values(), color=colors_summary, edgecolor='black', linewidth=1)
ax2.set_ylabel('Speedup Factor (x)', fontsize=12, fontweight='bold')
ax2.set_title('Performance Statistics', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for i, (k, v) in enumerate(summary_data.items()):
    ax2.text(i, v + 10, f'{v:.1f}x', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('performance_results.png', dpi=300, bbox_inches='tight')
print("✓ Chart saved as: performance_results.png")
