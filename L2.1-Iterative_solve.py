import math

_k = 100
_eps = 1e-10

x1_a = -1
x1_b = 0

x2_a = 0
x2_b = 3

x3_a = 3
x3_b = 6

F = lambda x: ((x ** 2) - (2 ** x))

def SearchRootBinary(func, x_left, x_right, eps):
    if func(x_left) * func(x_right) > 0:
        return None
    if func(x_left) == 0:
        return x_left
    if func(x_right) == 0:
        return x_right

    a = x_left
    b = x_right
    x = (a + b) / 2

    while (math.fabs(b - a) > 2 * eps):
        if (func(a) * func(x) < 0):
            b = x
        elif (func(b) * func(x) < 0):
            a = x
        else:
            if (func(x) == 0):
                return x
            if (func(a) == 0):
                return a
            if (func(b) == 0):
                return b
            b -= (b - a) / _k
            x = (a + b) / 2	

    return x

def SearchRootGallop(func, x_left, x_right, eps):
	if func(x_left) * func(x_right) > 0:
		return None
	if func(x_left) == 0:
		return x_left
	if func(x_right) == 0:
		return x_right
		
	step = 10 * eps
	
	middle = x_left + step
	while func(x_left) * func(middle) > 0 and middle < x_right:
		step = step * 2
		middle += step
	
	if middle < x_right:
		return SearchRootBinary(func, x_left, middle, eps)
	else:
		return SearchRootBinary(func, middle - step, x_right, eps)

func = lambda x: F(x)

Binary = [SearchRootBinary(func, x1_a, x1_b, _eps),
		  SearchRootBinary(func, x2_a, x2_b, _eps),
		  SearchRootBinary(func, x3_a, x3_b, _eps)]

Gallop = [SearchRootGallop(func, x1_a, x1_b, _eps),
		  SearchRootGallop(func, x2_a, x2_b, _eps),
		  SearchRootGallop(func, x3_a, x3_b, _eps)]

print("Результаты бинарного поиска:",Binary)
print("Результаты ускоряющегося поиска:",Gallop)