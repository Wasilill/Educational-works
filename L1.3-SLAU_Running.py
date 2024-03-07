import numpy as np
import math

A_1 = np.array([[-1, 3, 0, 0],
              [3, 4, -2, 0],
              [0, 2, 1, -1],
              [0, 0, 1, 2]], dtype = float)
b_1 = np.array([7, 3, 7, -3], dtype = float)

A_2 = np.array([[1, 4, 0, 0],
              [7, 9, 4, 0],
              [0, 5, -10, -3],
              [0, 0, 9, 1]], dtype = float)
b_2 = np.array([-15, 18, -119, 71], dtype = float)

def TripRibb(A):
    size = A[0].size
    k = np.zeros(shape=(3, size))    
    k[1][0] = A[0][0]
    for i in range(1, size):
        k[0][i] = A[i-1][i]
        k[1][i] = A[i][i]
        k[2][i] = A[i][i-1]    
    return k

def SolveSlaeThomas(A, b): 
    num = b.size
    k_a = np.zeros(num)
    k_b = np.zeros(num)

    k_a[0] = -A[0][1] / A[1][0] 
    k_b[0] = b[0] / A[1][0] 

    for i in range(1, num-1): 
        k = A[1][i] + A[2][i] * k_a[i - 1]
        k_a[i] = -A[0][i + 1] / k 
        k_b[i] = (b[i] - A[2][i] * k_b[i - 1]) / k

    k_b[num-1] = (b[num-1] - 
                  A[2][num-1] * 
                  k_b[num-2]) / (A[1][num-1] +
                                A[2][num-1] * k_a[num-2])

    result = np.zeros(num)    
    result[num-1] = k_b[num-1] 

    for i in range(num - 2, -1, -1): 
        result[i] = k_a[i] * result[i + 1] + k_b[i] 

    return result

print("Первая система: ", SolveSlaeThomas(TripRibb(A_1),b_1))
print("Вторая система: ", SolveSlaeThomas(TripRibb(A_2),b_2))
