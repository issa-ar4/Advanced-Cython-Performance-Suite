"""
Quick demo of the enhanced Cython features
"""

import numpy as np

print("🚀 Cython Performance Suite - Quick Demo")
print("="*60)

try:
    import algorithms_cy
    import algorithms_py
    
    print("\n✅ Successfully imported modules!")
    
    # Demo 1: Factorial
    print("\n1️⃣  FACTORIAL")
    print("-" * 40)
    n = 20
    cy_result = algorithms_cy.factorial(n)
    py_result = algorithms_py.factorial(n)
    print(f"factorial({n}) = {cy_result}")
    print(f"Results match: {cy_result == py_result}")
    
    # Demo 2: Fibonacci
    print("\n2️⃣  FIBONACCI")
    print("-" * 40)
    n = 30
    cy_result = algorithms_cy.fibonacci(n)
    py_result = algorithms_py.fibonacci(n)
    print(f"fibonacci({n}) = {cy_result}")
    print(f"Results match: {cy_result == py_result}")
    
    # Demo 3: Matrix Multiplication
    print("\n3️⃣  MATRIX MULTIPLICATION")
    print("-" * 40)
    size = 50
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    
    cy_result = algorithms_cy.matrix_multiply(A, B)
    py_result = algorithms_py.matrix_multiply(A, B)
    numpy_result = np.dot(A, B)
    
    print(f"Matrix size: {size}x{size}")
    print(f"Cython vs Python match: {np.allclose(cy_result, py_result)}")
    print(f"Cython vs NumPy match: {np.allclose(cy_result, numpy_result)}")
    
    # Demo 4: Parallel Matrix Multiplication
    print("\n4️⃣  PARALLEL MATRIX MULTIPLICATION")
    print("-" * 40)
    cy_parallel = algorithms_cy.matrix_multiply_parallel(A, B)
    print(f"Parallel vs sequential match: {np.allclose(cy_result, cy_parallel)}")
    
    # Demo 5: Sorting
    print("\n5️⃣  QUICKSORT")
    print("-" * 40)
    arr_size = 1000
    arr = np.random.randint(0, 1000, arr_size, dtype=np.int64)
    arr_copy = arr.copy()
    
    algorithms_cy.quicksort(arr, 0, len(arr) - 1)
    print(f"Sorted {arr_size} elements")
    print(f"Is sorted: {np.all(arr[:-1] <= arr[1:])}")
    print(f"First 10: {arr[:10]}")
    
    # Demo 6: Monte Carlo PI
    print("\n6️⃣  MONTE CARLO PI ESTIMATION")
    print("-" * 40)
    samples = 100000
    cy_pi = algorithms_cy.monte_carlo_pi(samples)
    py_pi = algorithms_py.monte_carlo_pi(samples)
    print(f"Samples: {samples:,}")
    print(f"Cython estimate: {cy_pi:.6f}")
    print(f"Python estimate: {py_pi:.6f}")
    print(f"Actual PI:       {np.pi:.6f}")
    print(f"Cython error:    {abs(cy_pi - np.pi):.6f}")
    
    # Demo 7: Prime Numbers
    print("\n7️⃣  PRIME NUMBERS")
    print("-" * 40)
    n = 1000
    cy_primes = algorithms_cy.primes_up_to(n)
    py_primes = algorithms_py.primes_up_to(n)
    print(f"Primes up to {n}: {len(cy_primes)}")
    print(f"Results match: {cy_primes == py_primes}")
    print(f"First 20 primes: {cy_primes[:20]}")
    
    # Demo 8: Levenshtein Distance
    print("\n8️⃣  LEVENSHTEIN DISTANCE")
    print("-" * 40)
    s1, s2 = "kitten", "sitting"
    cy_dist = algorithms_cy.levenshtein_distance(s1, s2)
    py_dist = algorithms_py.levenshtein_distance(s1, s2)
    print(f'Distance("{s1}", "{s2}") = {cy_dist}')
    print(f"Results match: {cy_dist == py_dist}")
    
    # Demo 9: Statistics
    print("\n9️⃣  STATISTICAL COMPUTATION")
    print("-" * 40)
    data = np.random.rand(10000)
    cy_stats = algorithms_cy.compute_statistics(data)
    py_stats = algorithms_py.compute_statistics(data)
    print(f"Data points: {len(data)}")
    print(f"Mean:     {cy_stats[0]:.6f}")
    print(f"Variance: {cy_stats[1]:.6f}")
    print(f"Std Dev:  {cy_stats[2]:.6f}")
    print(f"Results match: {np.allclose(cy_stats, py_stats)}")
    
    print("\n" + "="*60)
    print("✨ Demo complete! All algorithms working correctly.")
    print("\nNext steps:")
    print("  • Run 'python3 benchmark.py' for full performance comparison")
    print("  • Check *.html files for Cython optimization annotations")
    print("  • Read README.md for detailed documentation")
    print("="*60)
    
except ImportError as e:
    print(f"\n❌ Error: {e}")
    print("\n⚠️  Modules not found. Please build the project first:")
    print("   python3 setup.py build_ext --inplace")
    print("\nOr run the setup script:")
    print("   bash setup.sh")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
