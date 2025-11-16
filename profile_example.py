"""
Memory and performance profiling examples
Run with: python -m memory_profiler profile_example.py
"""

import numpy as np
from memory_profiler import profile


@profile
def profile_cython():
    """Profile Cython implementations"""
    import algorithms_cy
    
    # Test various algorithms
    result = algorithms_cy.factorial(100)
    fib = algorithms_cy.fibonacci(1000)
    
    A = np.random.rand(500, 500)
    B = np.random.rand(500, 500)
    C = algorithms_cy.matrix_multiply(A, B)
    
    arr = np.random.randint(0, 10000, 10000, dtype=np.int64)
    algorithms_cy.quicksort(arr, 0, len(arr) - 1)
    
    pi = algorithms_cy.monte_carlo_pi(1000000)
    primes = algorithms_cy.primes_up_to(10000)
    
    print(f"Factorial(100): {result}")
    print(f"Fibonacci(1000): {fib}")
    print(f"Estimated PI: {pi:.6f}")
    print(f"Number of primes up to 10,000: {len(primes)}")


@profile
def profile_python():
    """Profile Python implementations"""
    import algorithms_py
    
    # Test various algorithms
    result = algorithms_py.factorial(100)
    fib = algorithms_py.fibonacci(1000)
    
    A = np.random.rand(500, 500)
    B = np.random.rand(500, 500)
    C = algorithms_py.matrix_multiply(A, B)
    
    arr = np.random.randint(0, 10000, 10000, dtype=np.int64)
    algorithms_py.quicksort(arr, 0, len(arr) - 1)
    
    pi = algorithms_py.monte_carlo_pi(1000000)
    primes = algorithms_py.primes_up_to(10000)
    
    print(f"Factorial(100): {result}")
    print(f"Fibonacci(1000): {fib}")
    print(f"Estimated PI: {pi:.6f}")
    print(f"Number of primes up to 10,000: {len(primes)}")


if __name__ == "__main__":
    print("Profiling Cython implementation...")
    print("="*60)
    profile_cython()
    
    print("\n" + "="*60)
    print("Profiling Python implementation...")
    print("="*60)
    profile_python()
