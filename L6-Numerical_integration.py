import numpy as np
import matplotlib.pyplot as plt
import math

_zero = 1e-9
_n_0 = 6 # кол-во интервалов разбиения
_N = 8 # кол-во итераций

_TEXT = 8
_SPOT_NAME = 17

_NUM_AX = 100
_NUM_BASE = 25 # выбирать 6*k + 1 , k - цел.

_variant = 30
_FIRST = 0
_LAST = 1

_ANS = (10/np.log(10))

F = lambda x: ((10**x) + 1/np.log(10))

def tablex(x_left, x_right, num):
    return np.linspace(x_left,x_right,num)

def SolveSlaeGauss(_A, _b):
	A = np.array(_A)
	b = np.array(_b)
	_arg = b.size
	
	for i in range(_arg):
		max = 0.
		count = i

		for j in range(i,_arg):
			if (math.fabs(A[j][i]) > max):
				max = math.fabs(A[j][i])
				count = j

		temp = np.array(A[i])
		A[i] = A[count]
		A[count] = temp

		b[i], b[count] = b[count], b[i]

		if (math.fabs(A[i][i]) > _zero): 
			b[i] /= A[i][i]
		elif ((i == _arg - 1) & (math.fabs(b[i]) < _zero)):
			b[i] = 0.

		temp_i = 1
		while ((_arg - temp_i) >= i):
			if (math.fabs(A[i][i]) > _zero):
				A[i][_arg - temp_i] /= A[i][i]
			elif (i == _arg - 1):
				A[i][_arg - temp_i] = 0.
			temp_i += 1

		if (math.fabs(A[i][i]) > _zero):
			j = i + 1
			while(j < _arg):
				k = A[j][i] / A[i][i]

				temp_i = 1
				while((_arg - temp_i) >= i):
					A[j][_arg - temp_i] -= k * A[i][_arg - temp_i]
					temp_i+=1

				b[j] -= k * b[i]
				j += 1

	result = np.zeros(_arg)
	j = _arg - 1
	while(j >= 0):
		if (np.fabs(A[j][j]) >= _zero):
			result[j] = b[j]
			i = 1
			while(_arg - i > j):
				result[j] -= A[j][_arg - i] * result[_arg - i]
				i+=1
		else: 
			result[range(result.size)] = None
			return result

		j-=1

	return result

def LeastSquareMethod(x, xk, yk, _M):
	num = xk.size
	M = _M + 1

	C = np.zeros(shape = (M, M))
	b = np.zeros(shape = (M, M))
	h = np.zeros(shape = (M))

	temp_sum = np.zeros(2*M - 1)

	S = np.zeros(x.size)

	for i in range(2*M-1):	
		temp_sum[i] = np.sum(xk ** i)

	for j in range(M):
		h[j] = np.sum((xk ** j) * yk)
		for i in range(M):
			b[j][i] = temp_sum[i + j]

	C = SolveSlaeGauss(b,h)

	for i in range(M):
		S = S + C[i] * (x ** i)

	return S

def Integrate_L_R(x,y):
	sum = 0
	for i in range(x.size-1):
		sum += (x[i+1] - x[i]) * y[i]
	return sum

def IntegrateF_L_R(func, a, b, num):
	h = (b-a)/(num - 1)
	sum = 0
	x = a
	for i in range(num-1):
		sum += func(x)
		x += h
	sum *= h
	return sum

def Integrate_R_R(x,y):
	sum = 0
	for i in range(x.size-1):
		sum += (x[i+1] - x[i]) * y[i+1]
	return sum

def IntegrateF_R_R(func, a, b, num):
	h = (b-a)/(num - 1)
	sum = 0
	x = b
	for i in range(num - 1):
		sum += func(x)
		x -= h
	sum *= h
	return sum

def IntegrateF_M_R(func, a, b, num):
	h = (b-a)/(num - 1)
	sum = 0
	x = a
	for i in range(num - 1):
		sum += func(x+(h/2))
		x += h
	sum*=h
	return sum

def Integrate_T(x,y):
	sum = 0
	for i in range(x.size-1):
		sum += (x[i+1] - x[i]) * (y[i+1] + y[i])/2 
	return sum

def IntegrateF_T(func, a, b, num):
	h = (b-a)/(num - 1)
	sum = (func(a)+func(b))/2
	x = a + h
	for i in range(1,num-1):
		sum += func(x)
		x += h
	sum *= h
	return sum

def Integrate_S(x,y):
	h = x[1] - x[0]
	n_2 = (x.size - 1) // 2

	for i in range(1,x.size-1):
		if (x[i+1]-x[i] - h > _zero):
			print("Не работаю с отрезками разной длины")
			return None
	
	if (((x.size - 1) % 2) != 0):
		print("Не учитываю последний отрезок")

	sum = y[0]+y[-1]

	for i in range(1,n_2+1):
		sum += 4 * y[2*i - 1]

	for i in range(1,n_2):
		sum += 2 * y[2*i]

	sum *= h/3
	return sum

def IntegrateF_S(func, a, b, num):
	h = (b-a)/(num - 1)
	n_2 = (num - 1) // 2

	sum = (func(a) + func(b))

	for i in range(1,n_2+1):
		sum += 4 * func(a + (2*i - 1) * h)

	for i in range(1,n_2):
		sum += 2 * func(a + (2*i) * h)

	sum *= h / 3

	return sum

def Integrate_3_8(x,y):
	h = x[1] - x[0]
	N = x.size - 1
	for i in range(1,x.size-1):
		if (x[i+1]-x[i] - h > _zero):
			print("Не работаю с отрезками разной длины")
			return None
	
	if ((N % 3) != 0):
		print("Не учитываю последние отрезки")

	n_3 = N // 3

	sum = y[0] + y[-1]

	for i in range(1, n_3+1):
		sum += 3 * y[3*i - 2] + 3 * y[3*i - 1]

	for i in range(1, n_3):
		sum += 2 * y[3*i]
	
	sum *= 3/8 * h
	
	return sum

def IntegrateF_3_8(func, a, b, num):
	h = (b-a)/(num - 1)
	n_3 = (num - 1) // 3
	
	sum = func(a) + func(b)
	for i in range(1, n_3+1):
		sum += 3 * func(a + (3*i - 2) * h) + 3 * func(a + (3*i - 1) * h)
	for i in range(1, n_3):
		sum += 2 * func(a + 3*i*h)
	
	sum *= 3/8 * h
	
	return sum

def Rudge(func, a, b, N = _N):
	ln_J__J_h = np.zeros((6,N+1))
	x = np.zeros((N+1))

	for i in range(N + 1):
		num_i = ((_n_0 * (2 ** i)) + 1)
		ln_J__J_h[0][i] = math.log(math.fabs(_ANS - 
									   IntegrateF_L_R(func,a,b,num_i)))
		ln_J__J_h[1][i] = math.log(math.fabs(_ANS - 
									   IntegrateF_R_R(func,a,b,num_i)))
		ln_J__J_h[2][i] = math.log(math.fabs(_ANS - 
									   IntegrateF_M_R(func,a,b,num_i)))
		ln_J__J_h[3][i] = math.log(math.fabs(_ANS - 
									   IntegrateF_T(func,a,b,num_i)))
		ln_J__J_h[4][i] = math.log(math.fabs(_ANS - 
									   IntegrateF_S(func,a,b,num_i)))
		ln_J__J_h[5][i] = math.log(math.fabs(_ANS - 
									   IntegrateF_3_8(func,a,b,num_i)))
		x[i] = -i

	x = x * math.log1p(1)
	x_ax = tablex(x[N], x[0], _NUM_AX)

	fig, ax = plt.subplots(2,3)

	LSM_0 = LeastSquareMethod(x_ax, x, ln_J__J_h[0], 1)
	LSM_1 = LeastSquareMethod(x_ax, x, ln_J__J_h[1], 1)
	LSM_2 = LeastSquareMethod(x_ax, x, ln_J__J_h[2], 1)
	LSM_3 = LeastSquareMethod(x_ax, x, ln_J__J_h[3], 1)
	LSM_4 = LeastSquareMethod(x_ax, x, ln_J__J_h[4], 1)
	LSM_5 = LeastSquareMethod(x_ax, x, ln_J__J_h[5], 1)

	ax[0][0].set_title("Метод Левых Прямоугольников",fontsize = _TEXT)
	ax[0][1].set_title("Метод Правых Прямоугольников",fontsize = _TEXT)
	ax[0][2].set_title("Метод Средних Прямоугольников",fontsize = _TEXT)
	ax[1][0].set_title("Метод Трапеций",fontsize = _TEXT)
	ax[1][1].set_title("Метод Симпсона",fontsize = _TEXT)
	ax[1][2].set_title("Метод Ньютона \"3/8\"",fontsize = _TEXT)

	ax[0][0].plot(x_ax, LSM_0,'r',zorder=0)
	ax[0][0].scatter(x, ln_J__J_h[0], color = 'k', marker = 'o',zorder=1)

	ax[0][1].plot(x_ax, LSM_1,'m',zorder=0)
	ax[0][1].scatter(x, ln_J__J_h[1], color = 'k', marker = 'o',zorder=1)

	ax[0][2].plot(x_ax, LSM_2,'b',zorder=0)
	ax[0][2].scatter(x, ln_J__J_h[2], color = 'k', marker = 'o',zorder=1)

	ax[1][0].plot(x_ax, LSM_3,'g',zorder=0)
	ax[1][0].scatter(x, ln_J__J_h[3], color = 'k', marker = 'o',zorder=1)

	ax[1][1].plot(x_ax, LSM_4,'c',zorder=0)
	ax[1][1].scatter(x, ln_J__J_h[4], color = 'k', marker = 'o',zorder=1)

	ax[1][2].plot(x_ax, LSM_5,'y',zorder=0)
	ax[1][2].scatter(x, ln_J__J_h[5], color = 'k', marker = 'o',zorder=1)

	fig.set_figwidth(9)
	fig.set_figheight(6)

	plt.show()

func = lambda x: F(x)

x = tablex(_FIRST, _LAST, _NUM_BASE)
y = func(x)

a = _FIRST
b = _LAST
num = _NUM_BASE

LeftRect	 = Integrate_L_R(x,y)
LeftRectFunc = IntegrateF_L_R(func,a,b,num)
RightRect	  = Integrate_R_R(x,y)
RightRectFunc = IntegrateF_R_R(func,a,b,num)
MiddleRectFunc = IntegrateF_M_R(func,a,b,num)
Trapez = Integrate_T(x,y)
TrapezFunc = IntegrateF_T(func,a,b,num)
Simpson = Integrate_S(x,y)
SimpsonFunc = IntegrateF_S(func,a,b,num)
NewTone = Integrate_3_8(x,y)
NewToneFunc = IntegrateF_3_8(func,a,b,num)

print("Вариант #%d".center(_SPOT_NAME) %_variant,'\n')
print("Интеграл равен".center(_SPOT_NAME),_ANS,'\n',sep = '')

print("LeftRect = ".ljust(_SPOT_NAME), LeftRect, sep = '')
print("LeftRectFunc = ".ljust(_SPOT_NAME), LeftRectFunc,'\n', sep = '')

print("RightRect = ".ljust(_SPOT_NAME), RightRect, sep = '')
print("RightRectFunc = ".ljust(_SPOT_NAME), RightRectFunc,'\n', sep = '')

print("MiddleRectFunc = ".ljust(_SPOT_NAME), MiddleRectFunc,'\n', sep = '')

print("Trapez = ".ljust(_SPOT_NAME), Trapez, sep = '')
print("TrapezFunc = ".ljust(_SPOT_NAME), TrapezFunc,'\n', sep = '')

print("Simpson = ".ljust(_SPOT_NAME), Simpson, sep = '')
print("SimpsonFunc = ".ljust(_SPOT_NAME), SimpsonFunc,'\n', sep = '')

print("NewTone = ".ljust(_SPOT_NAME), NewTone, sep = '')
print("NewToneFunc = ".ljust(_SPOT_NAME), NewToneFunc,'\n', sep = '')

Rudge(func, a, b)
