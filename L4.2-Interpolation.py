import numpy as np
import matplotlib.pyplot as plt
import math

_TEXT = 16

_NUM_ANY = 50
_NUM_AX = 201
_NUM_10 = 10

_FIRST = -1
_LAST = 3

F = lambda x: (np.arctan(2*x+3))

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

fig, ax = plt.subplots(2,2)

func = lambda x:F(x)

x_ax = tablex(_FIRST, _LAST, _NUM_AX)
y_ax = func(x_ax)

x_10 = tablex(_FIRST, _LAST, _NUM_10)
y_10 = func(x_10)

x_any = tablex(_FIRST, _LAST, _NUM_ANY)
y_any = func(x_any)

LineSpline_10 = lspline(x_ax, x_10, y_10)
LineSpline_any = lspline(x_ax, x_any, y_any)

CubicSpline_10 = cspline(x_ax, x_10, y_10)
CubicSpline_any = cspline(x_ax, x_10, y_10)

ax[0][0].set_title("10 отрезков",fontsize = _TEXT)
ax[0][1].set_title(str(_NUM_ANY) + " отрезков",fontsize = _TEXT)

ax[0][0].set_ylabel("Линейный",fontsize = _TEXT)
ax[1][0].set_ylabel("Кубический",fontsize = _TEXT)

ax[0][0].plot(x_ax, LineSpline_10,'r')
ax[0][1].plot(x_ax, LineSpline_any,'b')

ax[1][0].plot(x_ax, CubicSpline_10,'r')
ax[1][1].plot(x_ax, CubicSpline_any,'b')

print("Значения в контрольных точках для 10 отрезков:\n", y_10)
print("Значения в контрольных точках для %d отрезков:\n" %_NUM_ANY, y_any)

plt.show()