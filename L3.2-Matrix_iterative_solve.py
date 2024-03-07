import math
import numpy as np

_eps = 1e-6
_k = 1e6*_eps
_zero = 1e-18

x0_1 = np.array([-2,0,0])
x0_2 = np.array([1,1,1])

func_1 = lambda x, y, z: 2*x + y + z
func_2 = lambda x, y, z: 3*x + 2*y + z
func_3 = lambda x, y, z: 3*((x+2)**3) + 2*((y+1)**3) + ((z+1)**3) - 27

df1_dx = lambda x, y, z: 2
df1_dy = lambda x, y, z: 1
df1_dz = lambda x, y, z: 1

df2_dx = lambda x, y, z: 3
df2_dy = lambda x, y, z: 2
df2_dz = lambda x, y, z: 1

df3_dx = lambda x, y, z: 9*((x+2)**2)
df3_dy = lambda x, y, z: 6*((y+1)**2)
df3_dz = lambda x, y, z: 3*((z+1)**2)

def System(x, y, z):
    result = np.zeros(3)

    result[0] = func_1(x, y, z)
    result[1] = func_2(x, y, z)
    result[2] = func_3(x, y, z)

    return result

def Jacobian(x, y, z):
	J = np.array(
		[[df1_dx(x,y,z), df1_dy(x,y,z), df1_dz(x,y,z)],
		[df2_dx(x,y,z), df2_dy(x,y,z), df2_dz(x,y,z)],
		[df3_dx(x,y,z), df3_dy(x,y,z), df3_dz(x,y,z)]])

	return J

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

def NewtonIter(Sy, x0, eps):
	root = np.array(x0)    
	prev_root = np.array([_k] * x0.size)
	prev_root = np.array(x0 + prev_root)
	delta = np.zeros(x0.size)

	temp = 0
	while (np.linalg.norm(root-prev_root)  >= eps):
		prev_root = root
		if (temp != 2):
			delta = SolveSlaeGauss(Jacobian(*root), (-Sy(*root)))
		root = root + delta
		temp = (temp + 1)%3

	return root

system = lambda x, y, z: System(x,y,z)

solution_1 = NewtonIter(system, x0_1, _eps)
solution_2 = NewtonIter(system, x0_2, _eps)

print("Возможные решения :")
print("1)",solution_1)
print("2)",solution_2)

if (max(system(*solution_1)) > (x0_1.size * _eps)):
	print("Первое решение не прошло проверку")
else:
	print("Первое решение прошло проверку")

if (max(system(*solution_2)) > (x0_1.size * _eps)):
	print("Второе решение не прошло проверку")
else:
	print("Второе решение прошло проверку")

