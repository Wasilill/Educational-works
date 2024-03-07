import math

_lambda = 0.01
_eps = 1e-12

x1_a = -3
x1_b = -2.2

x2_a = -2.1
x2_b = -1.7

x3_a = 0.99
x3_b = 2

x4_a = 2
x4_b = 3

F = lambda x: math.pow((-(x**5) + 4*(x**4) + 
                         2*(x**3) + 20*(x**2) + 24*x - 48),(1/6))

def SearchRootSimpleIterations(func,x_left,x_right,lambda_x,eps):
    x = (x_right + x_left) / 2
    x_prev = x + 10 * eps
    while(math.fabs(x - x_prev) > eps):
        x_prev = x
        x = x_prev - lambda_x * (func(x_prev) - x_prev)
    return x

func_plus = lambda x: F(x)
func_minus = lambda x: -F(x)

Root = [SearchRootSimpleIterations(func_minus, x1_a, x1_b, -_lambda, _eps),
		SearchRootSimpleIterations(func_minus, x2_a, x2_b, _lambda, _eps),
		SearchRootSimpleIterations(func_plus, x3_a, x3_b, _lambda, _eps),
        SearchRootSimpleIterations(func_plus, x4_a, x4_b, -_lambda, _eps)]

print("Корни данного уравнения:", Root)
