import timeit
import example_cy
import example_py

cy = timeit.timeit('example_cy.test(10000)', setup='import example_cy', number = 100)
py = timeit.timeit('example_py.test(10000)', setup='import example_py', number = 100)

print("Cython output: ", example_cy.test(12))
print("Python output: ", example_py.test(12))

print("Cython Runtime: ", cy)
print("Python Runtime: ", py)

print(f"Cython is {py/cy}x faster than Python")