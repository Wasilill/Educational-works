import numpy as np
import math

_zero = 1e-9

A_1=np.array([[4, -3, 4, -4],
			  [-2, -4, 0, -5],
			  [1, -1, 0, 1],
			  [-5, 2, -3, 2]], dtype = float)
b_1=np.array([-6, 10, -8, 12], dtype = float)

A_2=np.array([[-1, 0, 3, -2],
			  [-1, -1, -2, -5],
			  [1, -1, -7, -1],
			  [-2, -1, 2, -7]], dtype = float)
b_2=np.array([3, -10, -16, -7], dtype = float)

A_3=np.array([[1, 1, 0, 4],
			  [-4, -4, 3, -1],
			  [1, 1, -1, 4],
			  [-2, -2, 1, -3]], dtype = float)
b_3=np.array([1, -3, 1, 5], dtype = float)

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

print("Первая система: ", SolveSlaeGauss(A_1, b_1))
print("Вторая система: ", SolveSlaeGauss(A_2, b_2))
print("Третья система: ", SolveSlaeGauss(A_3, b_3))
