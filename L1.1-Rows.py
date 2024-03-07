import math

eps_1 = float (input("Введите первую отсечку\n"))
eps_2 = 1e-12
eps_3 = 1e-16

F_1 = lambda x: ((1 - 1.6 * 1e-5) ** x)
F_2 = lambda x: (math.cos(x) / ((x + 1) ** 2))
F_3 = lambda x: (((-1) ** x) * ((math.pi / 3) ** (2*x)) / math.factorial(2*x + 1))

def SumRow(func,N,eps):
    sum = 0
    temp = 10 * eps
    i = N 

    while(math.fabs(temp) >= eps):
        temp = func(i)
        sum += temp
        i += 1

    return sum

print("Сумма первого ряда при ",eps_1, " : ",SumRow(F_1,0,eps_1))
print("Сумма первого ряда при ",eps_2, " : ",SumRow(F_1,0,eps_2))
print("Сумма первого ряда при ",eps_3, " : ",SumRow(F_1,0,eps_3))

print("Сумма первого ряда при ",eps_1, " : ",SumRow(F_2,1,eps_1))

print("Сумма первого ряда при ",eps_1, " : ",SumRow(F_3,2,eps_1))