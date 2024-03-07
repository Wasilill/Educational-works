import numpy as np
import matplotlib.pyplot as plt
import math

_zero = 1e-9

_TEXT = 12

_NUM_AX = 100
_NUM_BASE = 10

_FIRST = 0
_LAST = 3

F = lambda x:((2*x-1)/(3*x+1) + np.sin(4*x-(math.pi/3)))

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

fig, ax = plt.subplots(1,3)

func = lambda x: F(x)

x_ax = tablex(_FIRST, _LAST, _NUM_AX)

x_base = tablex(_FIRST, _LAST, _NUM_BASE)
y_base = func(x_base)

pos_base = np.zeros(x_base.size, dtype = int)

for i in range(1,x_base.size-1):
	pos_base[i] = (np.array(x_ax[x_ax < x_base[i]])).size
	x_ax = np.insert(x_ax, pos_base[i],x_base[i])

pos_base[x_base.size-1] = _NUM_AX + _NUM_BASE - 3

ax[0].set_title("Линейная функция",fontsize = _TEXT)
ax[1].set_title("Параболическая функция",fontsize = _TEXT)
ax[2].set_title("Кубическая функция",fontsize = _TEXT)

LSM_M_1 = LeastSquareMethod(x_ax, x_base, y_base, 1)
LSM_M_2 = LeastSquareMethod(x_ax, x_base, y_base, 2)
LSM_M_3 = LeastSquareMethod(x_ax, x_base, y_base, 3)

ax[0].plot(x_ax, LSM_M_1,'r',zorder=0)
ax[0].scatter(x_base, y_base, color = 'b', marker = 'o',zorder=1)

ax[1].plot(x_ax, LSM_M_2,'r',zorder=0)
ax[1].scatter(x_base, y_base, color = 'b', marker = 'o',zorder=1)

ax[2].plot(x_ax, LSM_M_3,'r',zorder=0)
ax[2].scatter(x_base, y_base, color = 'b', marker = 'o',zorder=1)

output = np.zeros((8,x_base.size))

for i in range(x_base.size):
	output[0][i] = x_base[i]
	output[1][i] = func(x_base)[i]
	output[2][i] = LSM_M_1[pos_base[i]]
	output[3][i] = LSM_M_2[pos_base[i]]
	output[4][i] = LSM_M_3[pos_base[i]]
	output[5][i] = math.fabs(LSM_M_1[pos_base[i]] - y_base[i])
	output[6][i] = math.fabs(LSM_M_2[pos_base[i]] - y_base[i])
	output[7][i] = math.fabs(LSM_M_3[pos_base[i]] - y_base[i])

print("Значения x    ", end = '')
for i in output[0]:
	print('    ',"{0:+.4f}".format(i), end = '')
print()

print("Значения f(x) ", end = '')
for i in output[1]:
	print('    ',"{0:+.4f}".format(i),end = '')
print()

print("Значения ЛФ   ", end = '')
for i in output[2]:
	print('    ',"{0:+.4f}".format(i), end = '')
print()

print("Значения ПФ   ", end = '')
for i in output[3]:
	print('    ',"{0:+.4f}".format(i), end = '')
print()

print("Значения КФ   ", end = '')
for i in output[4]:
	print('    ',"{0:+.4f}".format(i), end = '')
print()

print("Невязки ЛФ    ", end = '')
for i in output[5]:
	print('    ',"{0:+.4f}".format(i), end = '')
print()

print("Невязки ПФ    ", end = '')
for i in output[6]:
	print('    ',"{0:+.4f}".format(i),end = '')
print()

print("Невязки КФ    ",end = '')
for i in output[7]:
	print('    ',"{0:+.4f}".format(i), end = '')
print()

print("Сумма квадратов невязок ЛФ = %.4f" % (output[5] * output[5]).sum(), end = '\n')
print("Сумма квадратов невязок ПФ = %.4f" % (output[6] * output[6]).sum(), end = '\n')
print("Сумма квадратов невязок КФ = %.4f" % (output[7] * output[7]).sum(), end = '\n')

plt.show()
