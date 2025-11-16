"""
Streamlined benchmarking suite
"""

import timeit
import numpy as np
import sys

# Import both implementations
import algorithms_cy
import algorithms_py

# Increase recursion limit for Python
sys.setrecursionlimit(10000)

print("\n" + "="*70)
print("🚀 CYTHON VS PYTHON PERFORMANCE BENCHMARK")
print("="*70)

results = []

# 1. Factorial
print("\n1️⃣  Factorial(20)")
print("-" * 70)
cy_time = timeit.timeit('algorithms_cy.factorial(20)', 'import algorithms_cy', number=10000)
py_time = timeit.timeit('algorithms_py.factorial(20)', 'import algorithms_py', number=10000)
speedup = py_time / cy_time
print(f"Cython: {cy_time:.6f}s | Python: {py_time:.6f}s | Speedup: {speedup:.2f}x")
results.append(('Factorial', speedup))

# 2. Fibonacci
print("\n2️⃣  Fibonacci(30)")
print("-" * 70)
cy_time = timeit.timeit('algorithms_cy.fibonacci(30)', 'import algorithms_cy', number=10000)
py_time = timeit.timeit('algorithms_py.fibonacci(30)', 'import algorithms_py', number=10000)
speedup = py_time / cy_time
print(f"Cython: {cy_time:.6f}s | Python: {py_time:.6f}s | Speedup: {speedup:.2f}x")
results.append(('Fibonacci', speedup))

# 3. Matrix Multiplication
print("\n3️⃣  Matrix Multiplication (100x100)")
print("-" * 70)
A = np.random.rand(100, 100)
B = np.random.rand(100, 100)
cy_time = timeit.timeit(lambda: algorithms_cy.matrix_multiply(A, B), number=10)
py_time = timeit.timeit(lambda: algorithms_py.matrix_multiply(A, B), number=10)
speedup = py_time / cy_time
print(f"Cython: {cy_time:.6f}s | Python: {py_time:.6f}s | Speedup: {speedup:.2f}x")
results.append(('Matrix Multiply', speedup))

# 4. Parallel vs Sequential
print("\n4️⃣  Matrix Multiplication - Parallel Speedup")
print("-" * 70)
seq_time = timeit.timeit(lambda: algorithms_cy.matrix_multiply(A, B), number=10)
par_time = timeit.timeit(lambda: algorithms_cy.matrix_multiply_parallel(A, B), number=10)
par_speedup = seq_time / par_time
print(f"Sequential: {seq_time:.6f}s | Parallel: {par_time:.6f}s | Speedup: {par_speedup:.2f}x")

# 5. Merge Sort
print("\n5️⃣  Merge Sort (5000 elements)")
print("-" * 70)
arr = np.random.randint(0, 10000, 5000, dtype=np.int64)
cy_time = timeit.timeit(lambda: algorithms_cy.merge_sort(arr), number=10)
py_time = timeit.timeit(lambda: algorithms_py.merge_sort(arr), number=10)
speedup = py_time / cy_time
print(f"Cython: {cy_time:.6f}s | Python: {py_time:.6f}s | Speedup: {speedup:.2f}x")
results.append(('Merge Sort', speedup))

# 6. Monte Carlo PI
print("\n6️⃣  Monte Carlo PI (100k samples)")
print("-" * 70)
cy_time = timeit.timeit('algorithms_cy.monte_carlo_pi(100000)', 'import algorithms_cy', number=10)
py_time = timeit.timeit('algorithms_py.monte_carlo_pi(100000)', 'import algorithms_py', number=10)
speedup = py_time / cy_time
print(f"Cython: {cy_time:.6f}s | Python: {py_time:.6f}s | Speedup: {speedup:.2f}x")
results.append(('Monte Carlo PI', speedup))

# 7. Simpson Integration
print("\n7️⃣  Simpson Integration (10k intervals)")
print("-" * 70)
cy_time = timeit.timeit('algorithms_cy.integrate_simpson(0.0, 3.14159, 10000)', 
                        'import algorithms_cy', number=100)
py_time = timeit.timeit('algorithms_py.integrate_simpson(0.0, 3.14159, 10000)', 
                        'import algorithms_py', number=100)
speedup = py_time / cy_time
print(f"Cython: {cy_time:.6f}s | Python: {py_time:.6f}s | Speedup: {speedup:.2f}x")
results.append(('Simpson Integration', speedup))

# 8. Statistics
print("\n8️⃣  Compute Statistics (10k elements)")
print("-" * 70)
data = np.random.rand(10000)
cy_time = timeit.timeit(lambda: algorithms_cy.compute_statistics(data), number=100)
py_time = timeit.timeit(lambda: algorithms_py.compute_statistics(data), number=100)
speedup = py_time / cy_time
print(f"Cython: {cy_time:.6f}s | Python: {py_time:.6f}s | Speedup: {speedup:.2f}x")
results.append(('Statistics', speedup))

# 9. Levenshtein Distance
print("\n9️⃣  Levenshtein Distance")
print("-" * 70)
cy_time = timeit.timeit('algorithms_cy.levenshtein_distance("kitten", "sitting")', 
                        'import algorithms_cy', number=1000)
py_time = timeit.timeit('algorithms_py.levenshtein_distance("kitten", "sitting")', 
                        'import algorithms_py', number=1000)
speedup = py_time / cy_time
print(f"Cython: {cy_time:.6f}s | Python: {py_time:.6f}s | Speedup: {speedup:.2f}x")
results.append(('Levenshtein', speedup))

# 10. Is Prime
print("\n🔟 Is Prime (large number)")
print("-" * 70)
cy_time = timeit.timeit('algorithms_cy.is_prime(982451653)', 'import algorithms_cy', number=1000)
py_time = timeit.timeit('algorithms_py.is_prime(982451653)', 'import algorithms_py', number=1000)
speedup = py_time / cy_time
print(f"Cython: {cy_time:.6f}s | Python: {py_time:.6f}s | Speedup: {speedup:.2f}x")
results.append(('Is Prime', speedup))

# 11. Primes Sieve
print("\n1️⃣1️⃣  Sieve of Eratosthenes (primes up to 10,000)")
print("-" * 70)
cy_time = timeit.timeit('algorithms_cy.primes_up_to(10000)', 'import algorithms_cy', number=100)
py_time = timeit.timeit('algorithms_py.primes_up_to(10000)', 'import algorithms_py', number=100)
speedup = py_time / cy_time
print(f"Cython: {cy_time:.6f}s | Python: {py_time:.6f}s | Speedup: {speedup:.2f}x")
results.append(('Primes Sieve', speedup))

# Summary
print("\n" + "="*70)
print("📊 SUMMARY")
print("="*70)

avg_speedup = sum(s for _, s in results) / len(results)
max_speedup = max(results, key=lambda x: x[1])
min_speedup = min(results, key=lambda x: x[1])

print(f"\n✨ Average Speedup: {avg_speedup:.2f}x")
print(f"🚀 Best Performance: {max_speedup[0]} ({max_speedup[1]:.2f}x)")
print(f"📈 Minimum Speedup: {min_speedup[0]} ({min_speedup[1]:.2f}x)")

print("\n🏆 All Benchmarks:")
for name, speedup in sorted(results, key=lambda x: x[1], reverse=True):
    bars = '█' * int(speedup / 10)
    print(f"  {name:25s} {speedup:7.2f}x {bars}")

print("\n" + "="*70)
print(f"✅ Cython is {avg_speedup:.1f}x faster on average!")
print("="*70 + "\n")
