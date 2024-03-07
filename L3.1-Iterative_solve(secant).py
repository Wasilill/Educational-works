import math

_eps = 1e-12

x1_a = 0.1
x1_b = 0.25

x2_a = 1
x2_b = 2

F = lambda x: ((x**2) - 2 - math.log(x))

def SearchRootSecant(func,x_0,x_1,eps):
    x_i_m = x_0
    x_i__ = x_1
    x_i_p = x_1 + 10 * eps
    while(math.fabs(x_i_p - x_i__) > eps):
        x_i_m = x_i__
        x_i__ = x_i_p
        k1 = func(x_i_m) * (x_i__ - x_i_m)
        k2 = func(x_i__) - func(x_i_m)
        x_i_p = x_i_m - (k1 / k2)
    return x_i_p

func = lambda x: F(x)

Root = [SearchRootSecant(func, x1_a, x1_b, _eps),
		SearchRootSecant(func, x2_a, x2_b, _eps)]

print("Корни данного уравнения:", Root)
