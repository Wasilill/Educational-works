import numpy as np
import matplotlib.pyplot as plt
import math

_TEXT = 16

_NUM_AX = 320

_FIRST = 0
_LAST  = 4

_NOS_1 = 20
_NOS_2 = 160

dy0 = lambda x, y0, y1 : y1
dy1 = lambda x, y0, y1 : (y0 * np.sin(x))

IV0 = 0
IV1 = 2

func = np.array([dy0, dy1],dtype = object)
initial_value = np.array([IV0, IV1],dtype = float)

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

def RungeKutta(func,x_first,x_last,num_of_steps,initial_value):
    length_of_steps = (x_last - x_first) / num_of_steps

    x_y = np.zeros(shape = (func.size + 1, num_of_steps + 1))
    x_y[0] = tablex(x_first, x_last, (num_of_steps + 1))
    for i in range(func.size):
        x_y[i+1][0] = initial_value[i]

    k_1 = np.zeros(shape = (func.size))
    k_2 = np.zeros(shape = (func.size))
    k_3 = np.zeros(shape = (func.size))
    x_temp = np.zeros(shape = 1)
    y_temp = np.zeros(shape = (func.size))

    for i in range(num_of_steps):
        x_temp = x_y[0,i]
        y_temp = x_y[1:,i]

        for j in range (func.size):
            k_1[j] = length_of_steps * func[j](x_temp,*y_temp)

        x_temp = x_y[0,i] + length_of_steps/3
        y_temp = x_y[1:,i] + k_1/3

        for j in range (func.size):
            k_2[j] = length_of_steps * func[j](x_temp,*y_temp)

        x_temp = x_y[0,i] + 2*length_of_steps/3
        y_temp = x_y[1:,i] + 2*k_2/3

        for j in range (func.size):
            k_3[j] = length_of_steps * func[j](x_temp,*y_temp)

        x_y[1:,i+1] = x_y[1:,i] + (k_1 + 3*k_3)/4

    return x_y

graph_1 = RungeKutta(func,_FIRST,_LAST,_NOS_1,initial_value)
graph_2 = RungeKutta(func,_FIRST,_LAST,_NOS_2,initial_value)

fig, ax = plt.subplots(2)

x_ax = tablex(_FIRST, _LAST, _NUM_AX)

x_1 = tablex(_FIRST, _LAST, _NOS_1+1)
x_2 = tablex(_FIRST, _LAST, _NOS_2+1)

LinSpline_1 = lspline(x_ax, x_1, graph_1[1])
LinSpline_2 = lspline(x_ax, x_2, graph_2[1])

CubSpline_1 = cspline(x_ax, x_1, graph_1[1])
CubSpline_2 = cspline(x_ax, x_2, graph_2[1])

ax[0].set_title("Линейный",fontsize = _TEXT)
ax[1].set_title("Кубический",fontsize = _TEXT)

ax[0].plot(x_ax, LinSpline_1,'b-',zorder=1, label = 'y(x), N = %d' %_NOS_1)
ax[0].plot(x_ax, LinSpline_2,'r-',zorder=1, label = 'y(x), N = %d' %_NOS_2)

ax[0].legend()

ax[1].plot(x_ax, CubSpline_1,'b-',zorder=1, label = 'y(x), N = %d' %_NOS_1)
ax[1].plot(x_ax, CubSpline_2,'r-',zorder=1, label = 'y(x), N = %d' %_NOS_2)

ax[1].legend()

plt.show()