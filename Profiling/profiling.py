# profiling.py
"""Python Essentials: Profiling.
<Name>
<Class>
<Date>
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
import os
from numba import jit
from time import time
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size



# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    
    for r in range(2,len(data)+1):
        for c in range(len(data[-r])):
            max_child = max(data[-r+1][c+1], data[-r+1][c])
            data[-r][c] += max_child
    return data[0][0]



# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = [2]
    current = 3
    while len(primes_list) < N:
        isprime = True
        for i in range(2,int(np.sqrt(current))+1):
            if current % i == 0:
                isprime = False
                break
        if isprime:
            primes_list.append(current)
        current += 2
    
    return primes_list


# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    return np.argmin(np.linalg.norm(A-x[:,np.newaxis], axis = 0))



# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alphabet = {ALPHA[a]:(ord(ALPHA[a])-64) for a in range(len(ALPHA))}
    #print(alphabet)
    with open(filename,'r') as infile:
        names = sorted(infile.read().replace('"','').split(','))
    
    return sum([sum([alphabet[name[i]] for i in range(len(name))]) * (index+1) for index,name in enumerate(names)])


# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    F_1 = 1
    yield F_1
    F_2 = 1
    yield F_2
    while True:
        F_1 = F_1 + F_2
        yield F_1
        F_2 = F_1 + F_2
        yield F_2

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    for i,f in enumerate(fibonacci()):
        if len(str(f)) >= N:
            return i+1


# Problem 6
def prime_sieve(N = 100000):
    """Yield all primes that are less than N."""
    numbers = np.arange(2,N)
    while len(numbers)>0:
        num_zero = numbers[0]
        mask = numbers % num_zero != 0
        numbers = numbers[mask]
        yield num_zero




# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product


def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    sizes = [i for i in range(1,50)]
    naive_power = []
    jit_power = []
    np_power = []
    for s in sizes:
        A = np.random.random((s,s))
        start = time()
        matrix_power(A,n)
        naive_power.append(time() - start)

        start = time()
        matrix_power_numba(A,n)
        jit_power.append(time() - start)

        start = time()
        np.linalg.matrix_power(A,n)
        np_power.append(time() - start)
    
    plt.plot(sizes, naive_power, 'r')
    plt.plot(sizes, jit_power, 'm')
    plt.plot(sizes, np_power, 'k')
    plt.title(f'A**{n} for various sizes')
    plt.xlabel('size')
    plt.ylabel('time')
    plt.legend(['naive_power', 'jit_power', 'np_power'])
    plt.show()


if __name__ == "__main__":
    os.chdir("/Users/chase/Desktop/Math345Volume1/byu_vol1/Profiling")
    
    # print(max_path_fast())

    # print(primes(20))
    # print(primes_fast(20))

    # A = np.array(   [[1,2,3],
    #                 [2,-3,4],
    #                 [3,4,5]])
                
    # x = np.array([-2,2,1])

    # print(nearest_column(A,x))
    # print(nearest_column_fast(A,x))


    # print(name_scores())
    # print(name_scores_fast())

    # print(fibonacci_digits())

    # for i,f in enumerate(fibonacci()):
    #     print(f)
    #     if i > 20:
    #         break

    # for i in prime_sieve():
    #     print(i)

    A = np.random.random((4,4))
    print(A)

    prob7()