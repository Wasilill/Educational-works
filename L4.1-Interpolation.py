import numpy as np
import matplotlib.pyplot as plt
import math

_NUM_AX = 100

_FIRST = -0.6
_LAST = 4.6

_H = 0.4

F = lambda x: np.power((5*x + 4), (1/3))

def tablex(x_left, x_right, step):
    x = np.array(x_left)
    temp = x_left
    while temp < (x_right - step):
        temp += step
        x = np.append(x,temp)
    x = np.append(x,x_right)
    return x

def GoGoLagrange(x, x_Go, L_Go):
    L = np.zeros(L_Go.size-1)
    h = x_Go.size - L.size

    for i in range(L.size):
        k = x_Go[i+h] - x_Go[i]
        line_1 = (x - x_Go[i]) * L_Go[i+1]
        line_2 = (x - x_Go[i+h]) * L_Go[i]
        L[i] = (line_1 - line_2) / k

    if (L.size != 1):
        return GoGoLagrange(x, x_Go, L)
    else:
        return L[0]

fig, ax = plt.subplots()

x_ax = np.linspace(_FIRST, _LAST, _NUM_AX)

x_base = tablex(_FIRST, _LAST, _H)
y_base = F(x_base)

y = np.array([GoGoLagrange(i, x_base, y_base) for i in x_ax])

ax = plt.plot(x_ax, y,'r',zorder=1)
ax = plt.scatter(x_base, y_base, color = 'b', marker = 'o',zorder=2)

print("Полученные значения в точках:\n",y)

plt.show()
