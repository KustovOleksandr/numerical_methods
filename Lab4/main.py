import math
import matplotlib.pyplot as plt

def M(x):
    return 50 * math.exp(-0.1 * x) + 5 * math.sin(x)

def derative_M(x):
    return -5 * math.exp(-0.1 * x) + 5 * math.cos(x)


x0 = 1.0
true_val = derative_M(x0)
print(f"Точне значення похідної в точці x0={x0}: {true_val}\n")

def approximation_derative(x, h):
    return (M(x + h) - M(x - h)) / (2 * h)

h_values = []
errors = []
best_h = 0
min_error = 10000000

power = -20
while power <= 3:
    h = 10 ** power
    approx_val = approximation_derative(x0, h)
    error = abs(approx_val - true_val)
    
    if error < min_error:
        min_error = error
        best_h = h
    
    h_values.append(h)

    if error == 0.0:
        error = 1e-20 
        
    errors.append(error)
    power = power + 1

print(f"Оптимальний крок h0: {best_h}")
print(f"Мінімальна похибка R0: {min_error}\n")

plt.figure(figsize=(9, 5)) 
plt.plot(h_values, errors, marker='o', linestyle='-')

plt.xscale('log')
plt.yscale('log')

plt.title('Залежність похибки від кроку h')
plt.xlabel('Крок h')
plt.ylabel('Похибка R')
plt.grid() 

h_fix = 0.001 

D_h = approximation_derative(x0, h_fix) #стандартний метод
R1 = abs(D_h - true_val)
print(f"1. Стандартний метод:")
print(f"   Похідна: {D_h}")
print(f"   Похибка R1: {R1}\n")


D_2h = approximation_derative(x0, 2 * h_fix) #метод рунге-ромберга
D_RR = D_h + (D_h - D_2h) / 3
R2 = abs(D_RR - true_val)
print(f"2. Метод Рунге-Ромберга:")
print(f"   Похідна: {D_RR}")
print(f"   Похибка R2: {R2}\n")

D_4h = approximation_derative(x0, 4 * h_fix) #метод ейткена
numerator = (D_2h ** 2) - (D_4h * D_h)
denominator = 2 * D_2h - (D_4h + D_h)
D_E = numerator / denominator
R3 = abs(D_E - true_val)
val = abs((D_4h - D_2h) / (D_2h - D_h))
p = (1 / math.log(2)) * math.log(val)
print(f"3. Метод Ейткена:")
print(f"   Похідна: {D_E}")
print(f"   Похибка R3: {R3}")
print(f"   Оцінка порядку точності p: {p}\n")

methods = ['Стандартний', 'Рунге-Ромберг', 'Ейткен']
method_errors = [R1, R2, R3]

plt.figure(figsize=(8, 5))
plt.bar(methods, method_errors)

plt.yscale('log') 

plt.title('Порівняння похибок різних методів')
plt.ylabel('Абсолютна похибка')
plt.grid()
plt.show()

print(f"РЕЖИМУ ПОЛИВУ")
drying_rate = D_RR
print(f"Поточна швидкість зміни вологості: {drying_rate:.4f}")

if drying_rate < -1.5:
    print("Грунт стрімко висихає.")
    print("Увімкнути полив")
elif drying_rate < 0:
    print("Грунт повільно висихає.")
    print("Полив поки не обов'язковий, перевірити через деякий час.")
else:
    print("Вологість зростає або стабільна.")
    print("Полив покі не потрібний")