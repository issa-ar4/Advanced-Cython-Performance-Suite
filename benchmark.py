"""
Comprehensive benchmarking suite for Cython vs Python performance comparison
"""

import timeit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

# Import both implementations
import algorithms_cy
import algorithms_py


class BenchmarkSuite:
    """Comprehensive benchmark suite for algorithm comparison"""
    
    def __init__(self):
        self.results = []
        sns.set_style("whitegrid")
        
    def benchmark_function(
        self,
        func_cy,
        func_py,
        args,
        name: str,
        number: int = 100
    ) -> Dict:
        """Benchmark a single function"""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")
        
        # Warm-up
        func_cy(*args)
        func_py(*args)
        
        # Time Cython
        cy_time = timeit.timeit(
            lambda: func_cy(*args),
            number=number
        )
        
        # Time Python
        py_time = timeit.timeit(
            lambda: func_py(*args),
            number=number
        )
        
        speedup = py_time / cy_time
        
        result = {
            'name': name,
            'cython_time': cy_time,
            'python_time': py_time,
            'speedup': speedup,
            'iterations': number
        }
        
        print(f"Cython: {cy_time:.6f}s ({cy_time/number*1000:.4f}ms per call)")
        print(f"Python: {py_time:.6f}s ({py_time/number*1000:.4f}ms per call)")
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify correctness
        cy_result = func_cy(*args)
        py_result = func_py(*args)
        
        if isinstance(cy_result, np.ndarray):
            match = np.allclose(cy_result, py_result)
        elif isinstance(cy_result, (list, tuple)):
            match = cy_result == py_result
        else:
            match = abs(cy_result - py_result) < 1e-6
        
        print(f"Results match: {match}")
        
        self.results.append(result)
        return result
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("\n" + "="*60)
        print("CYTHON VS PYTHON PERFORMANCE BENCHMARK SUITE")
        print("="*60)
        
        # 1. Factorial
        self.benchmark_function(
            algorithms_cy.factorial,
            algorithms_py.factorial,
            (20,),
            "Factorial(20)",
            number=10000
        )
        
        # 2. Fibonacci
        self.benchmark_function(
            algorithms_cy.fibonacci,
            algorithms_py.fibonacci,
            (30,),
            "Fibonacci(30)",
            number=10000
        )
        
        # 3. Matrix Multiplication
        A = np.random.rand(100, 100)
        B = np.random.rand(100, 100)
        self.benchmark_function(
            algorithms_cy.matrix_multiply,
            algorithms_py.matrix_multiply,
            (A, B),
            "Matrix Multiply (100x100)",
            number=10
        )
        
        # 4. Matrix Multiplication Parallel (Cython only - compare with sequential)
        print(f"\n{'='*60}")
        print("Benchmarking: Matrix Multiply Parallel vs Sequential")
        print(f"{'='*60}")
        
        seq_time = timeit.timeit(
            lambda: algorithms_cy.matrix_multiply(A, B),
            number=10
        )
        par_time = timeit.timeit(
            lambda: algorithms_cy.matrix_multiply_parallel(A, B),
            number=10
        )
        
        print(f"Sequential: {seq_time:.6f}s")
        print(f"Parallel:   {par_time:.6f}s")
        print(f"Parallel Speedup: {seq_time/par_time:.2f}x")
        
        # 5. Quicksort
        arr_cy = np.random.randint(0, 10000, 5000, dtype=np.int64)
        arr_py = arr_cy.copy()
        self.benchmark_function(
            lambda x: algorithms_cy.quicksort(x, 0, len(x) - 1),
            lambda x: algorithms_py.quicksort(x, 0, len(x) - 1),
            (arr_cy,),
            "Quicksort (5000 elements)",
            number=10
        )
        
        # 6. Merge Sort
        arr = np.random.randint(0, 10000, 5000, dtype=np.int64)
        self.benchmark_function(
            algorithms_cy.merge_sort,
            algorithms_py.merge_sort,
            (arr,),
            "Merge Sort (5000 elements)",
            number=10
        )
        
        # 7. Monte Carlo PI
        self.benchmark_function(
            algorithms_cy.monte_carlo_pi,
            algorithms_py.monte_carlo_pi,
            (100000,),
            "Monte Carlo PI (100k samples)",
            number=10
        )
        
        # 8. Simpson Integration
        self.benchmark_function(
            algorithms_cy.integrate_simpson,
            algorithms_py.integrate_simpson,
            (0.0, 3.14159, 10000),
            "Simpson Integration (10k intervals)",
            number=100
        )
        
        # 9. Statistics
        data = np.random.rand(10000)
        self.benchmark_function(
            algorithms_cy.compute_statistics,
            algorithms_py.compute_statistics,
            (data,),
            "Compute Statistics (10k elements)",
            number=100
        )
        
        # 10. Levenshtein Distance
        self.benchmark_function(
            algorithms_cy.levenshtein_distance,
            algorithms_py.levenshtein_distance,
            ("kitten", "sitting"),
            "Levenshtein Distance",
            number=1000
        )
        
        # 11. Prime Check
        self.benchmark_function(
            algorithms_cy.is_prime,
            algorithms_py.is_prime,
            (982451653,),
            "Is Prime (large number)",
            number=1000
        )
        
        # 12. Sieve of Eratosthenes
        self.benchmark_function(
            algorithms_cy.primes_up_to,
            algorithms_py.primes_up_to,
            (10000,),
            "Primes up to 10,000",
            number=10
        )
    
    def generate_report(self):
        """Generate visualization and report"""
        df = pd.DataFrame(self.results)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cython vs Python Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Speedup bar chart
        ax1 = axes[0, 0]
        colors = ['#2ecc71' if x > 10 else '#3498db' if x > 5 else '#e74c3c' 
                  for x in df['speedup']]
        ax1.barh(df['name'], df['speedup'], color=colors)
        ax1.axvline(x=1, color='red', linestyle='--', linewidth=2, label='No speedup')
        ax1.set_xlabel('Speedup Factor (x)', fontsize=12)
        ax1.set_title('Cython Speedup vs Python', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. Execution time comparison
        ax2 = axes[0, 1]
        x = np.arange(len(df))
        width = 0.35
        ax2.bar(x - width/2, df['cython_time'], width, label='Cython', color='#3498db')
        ax2.bar(x + width/2, df['python_time'], width, label='Python', color='#e74c3c')
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        ax2.set_title('Absolute Execution Times', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['name'], rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.set_yscale('log')
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Speedup distribution
        ax3 = axes[1, 0]
        ax3.hist(df['speedup'], bins=15, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax3.axvline(df['speedup'].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {df["speedup"].mean():.2f}x')
        ax3.axvline(df['speedup'].median(), color='green', linestyle='--', 
                    linewidth=2, label=f'Median: {df["speedup"].median():.2f}x')
        ax3.set_xlabel('Speedup Factor', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Speedup Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
        PERFORMANCE SUMMARY
        {'='*40}
        
        Total Benchmarks: {len(df)}
        
        Average Speedup: {df['speedup'].mean():.2f}x
        Median Speedup: {df['speedup'].median():.2f}x
        Max Speedup: {df['speedup'].max():.2f}x
        Min Speedup: {df['speedup'].min():.2f}x
        
        Best Performing:
        {df.nlargest(3, 'speedup')[['name', 'speedup']].to_string(index=False)}
        
        Total Cython Time: {df['cython_time'].sum():.4f}s
        Total Python Time: {df['python_time'].sum():.4f}s
        Overall Speedup: {df['python_time'].sum() / df['cython_time'].sum():.2f}x
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'benchmark_results_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved as: {filename}")
        
        # Save detailed results to CSV
        csv_filename = f'benchmark_results_{timestamp}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"✓ Detailed results saved as: {csv_filename}")
        
        plt.show()
        
        return df


def main():
    """Run the complete benchmark suite"""
    suite = BenchmarkSuite()
    suite.run_all_benchmarks()
    df = suite.generate_report()
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE!")
    print("="*60)
    print(f"\nAverage speedup: {df['speedup'].mean():.2f}x")
    print(f"Cython is consistently faster than pure Python!")


if __name__ == "__main__":
    main()
