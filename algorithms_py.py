"""
Pure Python implementations of various algorithms
For benchmarking against Cython versions
"""

import numpy as np
from math import sqrt, sin


# ============================================================================
# FACTORIAL AND COMBINATORICS
# ============================================================================

def factorial(n):
    """Compute factorial"""
    if n < 0:
        return -1
    if n == 0 or n == 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def fibonacci(n):
    """Compute nth Fibonacci number"""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


# ============================================================================
# MATRIX OPERATIONS
# ============================================================================

def matrix_multiply(A, B):
    """Matrix multiplication"""
    m, n = A.shape
    p = B.shape[1]
    C = np.zeros((m, p))
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C


# ============================================================================
# SORTING ALGORITHMS
# ============================================================================

def quicksort(arr, low, high):
    """In-place quicksort"""
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort(arr, low, pivot_index - 1)
        quicksort(arr, pivot_index + 1, high)


def partition(arr, low, high):
    """Partition for quicksort"""
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def merge_sort(arr):
    """Merge sort"""
    result = arr.copy()
    merge_sort_helper(result, 0, len(result) - 1)
    return result


def merge_sort_helper(arr, left, right):
    """Recursive helper for merge sort"""
    if left < right:
        mid = (left + right) // 2
        merge_sort_helper(arr, left, mid)
        merge_sort_helper(arr, mid + 1, right)
        merge(arr, left, mid, right)


def merge(arr, left, mid, right):
    """Merge two sorted subarrays"""
    L = arr[left:mid + 1].copy()
    R = arr[mid + 1:right + 1].copy()
    
    i = j = 0
    k = left
    
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
    
    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1
    
    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1


# ============================================================================
# NUMERICAL COMPUTATIONS
# ============================================================================

def monte_carlo_pi(n_samples):
    """Estimate PI using Monte Carlo"""
    import random
    random.seed(42)
    inside = 0
    
    for _ in range(n_samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside += 1
    
    return 4.0 * inside / n_samples


def integrate_simpson(a, b, n):
    """Simpson's rule for integration"""
    h = (b - a) / n
    result = sin(a) + sin(b)
    
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            result += 2 * sin(x)
        else:
            result += 4 * sin(x)
    
    return result * h / 3.0


def compute_statistics(data):
    """Compute mean, variance, and standard deviation"""
    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = sqrt(variance)
    return np.array([mean, variance, std_dev])


# ============================================================================
# STRING PROCESSING
# ============================================================================

def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance"""
    m, n = len(s1), len(s2)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost
            )
    
    return d[m][n]


# ============================================================================
# PRIME NUMBERS
# ============================================================================

def is_prime(n):
    """Check if number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    
    return True


def primes_up_to(n):
    """Sieve of Eratosthenes"""
    is_prime_arr = [True] * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    
    for i in range(2, int(sqrt(n)) + 1):
        if is_prime_arr[i]:
            j = i * i
            while j <= n:
                is_prime_arr[j] = False
                j += i
    
    return [i for i in range(2, n + 1) if is_prime_arr[i]]
