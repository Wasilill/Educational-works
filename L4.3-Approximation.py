import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.optimize import *

C = 1.03439984
T = 1.75418438
g = 9.8

def composite_simpson(a, b, n, f):
    if n % 2 != 0:
        n += 1
    x = np.linspace(a, b, n+1)
    h = (b - a)/n
    return h/3. * (f(x[0]) +
            2 * np.sum([f(x_i) for x_i in x[2:-1:2]]) +
            4 * np.sum([f(x_i) for x_i in x[1::2]]) +
            f(x[-1]))


def composite_trapezoid(a, b, n, f):
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    return h/2. * (f(x[0]) +
            2 * np.sum([f(x_i) for x_i in x[1:-1:1]]) +
            f(x[-1]))

def f_ex(x):
    return x**2
# print(composite_simpson(0, 5, 10, f_ex))
# print(composite_trapezoid(0, 5, 10, f_ex))


def xt(t, x):
    return C*t - C*1/2*sin(2*t) -x

def tx(x):
    return fsolve(xt,0, x)

def yx(x):
    t = tx(x)
    return C*(1/2 - 1/2*cos(2*t))

def dyx(x, dx = 0.001):
    return (yx(x+dx) - yx(x))/dx

def Fy(x):
    return np.sqrt((1 + (dyx(x)**2))/(2 * 9.8 * yx(x)))

def real_graphs(a, b):

    n = np.logspace(np.log10(9999), np.log10(3), 50, dtype=int)
    h = np.array((b - a) / n)
    hfs = np.logspace(-3, -1, 4)

    cs = [composite_simpson(a, b, n_, Fy) for n_ in n]
    ct = [composite_trapezoid(a, b, n_, Fy) for n_ in n]

    plt.loglog(h, cs, 'b.', label = "Simpson")
    plt.loglog(h, ct, 'r.', label = "trapezoid")
    plt.xlim(0.95,0.0001)


def err_graphs(a, b):

    n = np.logspace(np.log10(9999), np.log10(3), 50, dtype=int)
    h = np.array((b - a) / n)
    hfs = np.logspace(-3, -1, 4)

    absol = composite_simpson(a, b, 11000, Fy)

    err_cs = [np.abs(absol - composite_simpson(a, b, n_, Fy)) for n_ in n]
    err_ct = [np.abs(absol - composite_trapezoid(a, b, n_, Fy)) for n_ in n]

    plt.loglog(h, err_cs, 'b-', label ="Simpson_err")
    plt.loglog(h, err_ct,'r-', label = "trapezoid_err")
    plt.loglog(hfs, [(i**4) for i in hfs], 'k:', label = r"$O(h^4)$")
    plt.loglog(hfs, [(i**2) for i in hfs],'k--', label = r"$O(h^2)$")
    plt.ylim(0.45,1.7e-15)


def linear_func_coeffs(N, A, B, f):
    coeffs = np.zeros((N - 1, 3))
    x = np.linspace(A, B, N)
    y0 = 0
    ysh = 1
    x0 = 2
    for i in range(0, N - 1):
        coeffs[i][y0] = f(x[i])
        coeffs[i][ysh] = (f(x[i + 1]) - f(x[i])) / (x[i + 1] - x[i])
        coeffs[i][x0] = x[i]
    return coeffs


def yx_discr(x, j, coeffs):
    y0 = 0
    ysh = 1
    x0 = 2
    return coeffs[j][y0] + coeffs[j][ysh] * (x - coeffs[j][x0])


def Flsy(x, j, coeffs):
    return np.sqrt((1 + ((coeffs[j][1]) ** 2)) / (2 * g * yx_discr(x, j, coeffs)))

def pr_lin(A, B, N):
    x = np.linspace(A, B, N)
    coeffs = linear_func_coeffs(N, A, B, yx)
    fx = []
    fy = []
    for i in range(0, N - 1):
        fx.append(x[i])
        fy.append(yx_discr(x[i], i, coeffs))
        # fx.append((x[i+1]+x[i])/2)
        # fy.append(yx_discr((x[i+1]+x[i])/2, i, coeffs))
    plt.plot(fx, fy, 'ko')
    plt.plot(fx, fy, 'b-')


a = 0.001
b = 2

real_graphs(a, b)             # Графики не учитывая погрешность
err_graphs(a, b)              # Графики учитывая погрешность
pr_lin(a, b, 10)              # Кусочно-линейная интерполяция

plt.legend()
plt.show()
