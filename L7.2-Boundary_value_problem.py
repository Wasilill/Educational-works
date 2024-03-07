import numpy as np
import matplotlib.pyplot as plt
import math

_TEXT = 16

_NUM_AX = 100

_FIRST = 0
_LAST = 1

_NOS_1 = 10
_NOS_2 = 50

p = lambda x : -6
q = lambda x : 8
f = lambda x : 10

A = 1
B = 2

PRM = np.array([p, q, f],dtype = object)
IV = np.array([A,B])

def tablex(x_left, x_right, num):
    return np.linspace(x_left,x_right,num)

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

def lspline(x,xk, y):
    num = xk.size
    S = np.array([y[0]])

    L = np.zeros(num)
    F = np.zeros(num)

    for i in range(1, num):
        L[i] = (y[i] - y[i-1]) / (xk[i] - xk[i-1])
        F[i] = y[i-1] - L[i] * xk[i-1]

        x_temp = np.array(x[xk[i-1] < x])
        x_temp = np.array(x_temp[x_temp <= xk[i]])
        S_temp = np.array(L[i] * x_temp + F[i])
        S = np.concatenate((S,S_temp), axis = 0)

    return S

def cspline(x_axes, x, y):
    num = x.size
    h = np.zeros((num))

    c = np.zeros((num-2,num))
    f_h = np.zeros((num-2))

    B = np.zeros((num))
    A = np.zeros((num))
    D = np.zeros((num))
    
    for i in range(1,num):  
        h[i] = x[i]-x[i-1]

    for i in range(1,num-1):
        c[i-1][i-1] = h[i]/3
        c[i-1][i] = 2*(h[i+1]+h[i])/3
        c[i-1][i+1] = h[i+1]/3

        f_h[i-1] = (y[i+1] - 
                    y[i])/(h[i+1]) - (y[i] - 
                                      y[i-1])/(h[i])
        
    c = np.delete(c,0,axis=1)
    c = np.delete(c,-1,axis=1)

    C = np.array(SolveSlaeThomas(TripRibb(c),f_h))
    C = np.concatenate((C,[0]), axis = 0)
    C = np.concatenate(([0],C), axis = 0)
    C = np.concatenate(([0],C), axis = 0)

    for i in range(1,num):
        A[i] = y[i-1]
        B[i] = (y[i] - y[i-1])/(h[i]) - (2*C[i]+C[i+1])*(h[i]/3)        
        D[i] = (C[i+1]-C[i])/(3*h[i])

    S = np.array([A[1]])

    for i in range(1,num):
        x_s = np.array(x_axes[x[i-1] < x_axes])
        x_s = np.array(x_s[x_s <= x[i]])
        S_x = np.array((D[i] * ((x_s - x[i-1]) ** 3)) + 
                       (C[i] * ((x_s - x[i-1]) ** 2)) + 
                       (B[i] * (x_s - x[i-1])) + A[i])
        S = np.concatenate((S,S_x), axis = 0)

    return S

def L_H_D_E_2ord(x_first,x_last,num_of_steps,parameters,initial_value):
    h = (x_last - x_first) / num_of_steps

    x = tablex(x_first, x_last, (num_of_steps + 1))
    p = np.array([])
    q = np.array([])
    f = np.array([])

    for i in range(x.size - 2):
        p = np.append(p, PRM[0](x[i]))
        q = np.append(q, PRM[1](x[i]))
        f = np.append(f, PRM[2](x[i]))

    A = np.zeros(shape = (x.size - 2,x.size - 2))
    b = np.zeros(shape = (x.size - 2))

    A[0][0] = (h ** 2) * q[0] - 2
    A[0][1] = 1 + (h / 2) * p[0]
    b[0] = (h ** 2) * f[0] - initial_value[0] * (1 - (h / 2) * p[0])

    for i in range(1,x.size - 3):
        A[i][i-1] = 1 - (h / 2) * p[i]
        A[i][i]   = (h ** 2) * q[i] - 2
        A[i][i+1] = 1 + (h / 2) * p[i]

        b[i] = (h ** 2) * f[i]

    A[x.size - 3][x.size - 4] = 1 - (h / 2) * p[x.size - 3]
    A[x.size - 3][x.size - 3] = (h ** 2) * q[x.size - 3] - 2
    b[x.size - 3] = (h ** 2) * f[x.size - 3] - initial_value[1] * (1 + (h / 2) * p[x.size - 3])

    y = np.array(initial_value[0])
    y = np.append(y, SolveSlaeThomas(TripRibb(A),b))
    y = np.append(y, initial_value[1])

    return y

graph_1 = L_H_D_E_2ord(_FIRST,_LAST,_NOS_1,PRM,IV)
graph_2 = L_H_D_E_2ord(_FIRST,_LAST,_NOS_2,PRM,IV)

fig, ax = plt.subplots()

x_ax = tablex(_FIRST, _LAST, _NUM_AX)
x_1 = tablex(_FIRST, _LAST, _NOS_1+1)
x_2 = tablex(_FIRST, _LAST, _NOS_2+1)

LinSpline_1 = lspline(x_ax, x_1, graph_1)
LinSpline_2 = lspline(x_ax, x_2, graph_2)
CubSpline = cspline(x_ax, x_1, graph_1)

ax.plot(x_ax, LinSpline_1,'r-',zorder=1, label = "Линейный y(x), N = %d" %_NOS_1)
ax.plot(x_ax, LinSpline_2,'k-',zorder=1, label = "Линейный y(x), N = %d" %_NOS_2)
ax.plot(x_ax, CubSpline,'b-',zorder=2, label = "Кубический y(x), N = %d" %_NOS_1)

ax.legend()

plt.show()
