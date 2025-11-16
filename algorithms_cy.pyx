# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Advanced Cython implementations of various algorithms
Optimized for maximum performance
"""

from libc.math cimport sqrt, sin, cos, exp, log
from libc.stdlib cimport malloc, free
from cython.parallel import prange
cimport numpy as np
import numpy as np

# ============================================================================
# FACTORIAL AND COMBINATORICS
# ============================================================================

cpdef long long factorial(int n) nogil:
    """Compute factorial using C types and nogil for maximum speed"""
    cdef long long result = 1
    cdef int i
    
    if n < 0:
        return -1
    if n == 0 or n == 1:
        return 1
    
    for i in range(2, n + 1):
        result *= i
    
    return result


cpdef long long fibonacci(int n) nogil:
    """Compute nth Fibonacci number using iterative approach"""
    cdef long long a = 0, b = 1, temp
    cdef int i
    
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    for i in range(2, n + 1):
        temp = a + b
        a = b
        b = temp
    
    return b


# ============================================================================
# MATRIX OPERATIONS
# ============================================================================

cpdef np.ndarray[double, ndim=2] matrix_multiply(
    double[:, :] A,
    double[:, :] B
):
    """Optimized matrix multiplication using typed memoryviews"""
    cdef int i, j, k
    cdef int m = A.shape[0]
    cdef int n = A.shape[1]
    cdef int p = B.shape[1]
    cdef double temp
    
    cdef np.ndarray[double, ndim=2] C = np.zeros((m, p), dtype=np.float64)
    cdef double[:, :] C_view = C
    
    for i in range(m):
        for j in range(p):
            temp = 0.0
            for k in range(n):
                temp += A[i, k] * B[k, j]
            C_view[i, j] = temp
    
    return C


cpdef np.ndarray[double, ndim=2] matrix_multiply_parallel(
    double[:, :] A,
    double[:, :] B
):
    """Parallel matrix multiplication using OpenMP"""
    cdef int i, j, k
    cdef int m = A.shape[0]
    cdef int n = A.shape[1]
    cdef int p = B.shape[1]
    
    cdef np.ndarray[double, ndim=2] C = np.zeros((m, p), dtype=np.float64)
    cdef double[:, :] C_view = C
    
    for i in prange(m, nogil=True):
        for j in range(p):
            C_view[i, j] = 0.0
            for k in range(n):
                C_view[i, j] += A[i, k] * B[k, j]
    
    return C


# ============================================================================
# SORTING ALGORITHMS
# ============================================================================

cpdef void quicksort(long[:] arr, int low, int high) noexcept nogil:
    """In-place quicksort implementation"""
    cdef int pivot_index
    
    if low < high:
        pivot_index = partition(arr, low, high)
        quicksort(arr, low, pivot_index - 1)
        quicksort(arr, pivot_index + 1, high)


cdef int partition(long[:] arr, int low, int high) nogil:
    """Partition function for quicksort"""
    cdef long pivot = arr[high]
    cdef int i = low - 1
    cdef int j
    cdef long temp
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
    
    temp = arr[i + 1]
    arr[i + 1] = arr[high]
    arr[high] = temp
    
    return i + 1


cpdef np.ndarray[long, ndim=1] merge_sort(long[:] arr):
    """Merge sort implementation returning sorted array"""
    cdef int n = arr.shape[0]
    cdef np.ndarray[long, ndim=1] result = np.array(arr, dtype=np.int64)
    cdef long[:] result_view = result
    
    merge_sort_helper(result_view, 0, n - 1)
    return result


cdef void merge_sort_helper(long[:] arr, int left, int right) noexcept nogil:
    """Recursive helper for merge sort"""
    cdef int mid
    
    if left < right:
        mid = (left + right) // 2
        merge_sort_helper(arr, left, mid)
        merge_sort_helper(arr, mid + 1, right)
        merge(arr, left, mid, right)


cdef void merge(long[:] arr, int left, int mid, int right) noexcept nogil:
    """Merge two sorted subarrays"""
    cdef int n1 = mid - left + 1
    cdef int n2 = right - mid
    cdef long* L = <long*>malloc(n1 * sizeof(long))
    cdef long* R = <long*>malloc(n2 * sizeof(long))
    cdef int i, j, k
    
    for i in range(n1):
        L[i] = arr[left + i]
    
    for j in range(n2):
        R[j] = arr[mid + 1 + j]
    
    i = 0
    j = 0
    k = left
    
    while i < n1 and j < n2:
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1
    
    while i < n1:
        arr[k] = L[i]
        i += 1
        k += 1
    
    while j < n2:
        arr[k] = R[j]
        j += 1
        k += 1
    
    free(L)
    free(R)


# ============================================================================
# NUMERICAL COMPUTATIONS
# ============================================================================

cpdef double monte_carlo_pi(int n_samples) nogil:
    """Estimate PI using Monte Carlo method"""
    cdef int i, inside = 0
    cdef double x, y, distance
    cdef unsigned int seed = 42
    
    for i in range(n_samples):
        # Simple LCG random number generator
        seed = (1103515245 * seed + 12345) & 0x7fffffff
        x = (<double>seed) / 2147483647.0
        
        seed = (1103515245 * seed + 12345) & 0x7fffffff
        y = (<double>seed) / 2147483647.0
        
        distance = x * x + y * y
        if distance <= 1.0:
            inside += 1
    
    return 4.0 * inside / n_samples


cpdef double integrate_simpson(double a, double b, int n):
    """Simpson's rule for numerical integration of sin(x)"""
    cdef double h = (b - a) / n
    cdef double result = sin(a) + sin(b)
    cdef int i
    cdef double x
    
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            result += 2 * sin(x)
        else:
            result += 4 * sin(x)
    
    return result * h / 3.0


cpdef np.ndarray[double, ndim=1] compute_statistics(double[:] data):
    """Compute mean, variance, and standard deviation in one pass"""
    cdef int n = data.shape[0]
    cdef double mean = 0.0, variance = 0.0, std_dev
    cdef int i
    
    # Compute mean
    for i in range(n):
        mean += data[i]
    mean /= n
    
    # Compute variance
    for i in range(n):
        variance += (data[i] - mean) ** 2
    variance /= n
    std_dev = sqrt(variance)
    
    cdef np.ndarray[double, ndim=1] result = np.array([mean, variance, std_dev])
    return result


# ============================================================================
# STRING PROCESSING
# ============================================================================

cpdef int levenshtein_distance(str s1, str s2):
    """Compute Levenshtein distance between two strings"""
    cdef int m = len(s1)
    cdef int n = len(s2)
    cdef int i, j
    cdef int cost
    
    # Create distance matrix
    cdef np.ndarray[long, ndim=2] d = np.zeros((m + 1, n + 1), dtype=np.int64)
    cdef long[:, :] d_view = d
    
    for i in range(m + 1):
        d_view[i, 0] = i
    
    for j in range(n + 1):
        d_view[0, j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d_view[i, j] = min(
                d_view[i - 1, j] + 1,      # deletion
                d_view[i, j - 1] + 1,      # insertion
                d_view[i - 1, j - 1] + cost # substitution
            )
    
    return d_view[m, n]


# ============================================================================
# PRIME NUMBERS
# ============================================================================

cpdef bint is_prime(long n) nogil:
    """Check if a number is prime"""
    cdef long i
    
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


cpdef list primes_up_to(int n):
    """Sieve of Eratosthenes to find all primes up to n"""
    cdef np.ndarray[unsigned char, ndim=1] is_prime_arr = np.ones(n + 1, dtype=np.uint8)
    cdef unsigned char[:] is_prime_view = is_prime_arr
    cdef int i, j
    cdef list result = []
    
    is_prime_view[0] = 0
    is_prime_view[1] = 0
    
    for i in range(2, <int>sqrt(n) + 1):
        if is_prime_view[i]:
            j = i * i
            while j <= n:
                is_prime_view[j] = 0
                j += i
    
    for i in range(2, n + 1):
        if is_prime_view[i]:
            result.append(i)
    
    return result
