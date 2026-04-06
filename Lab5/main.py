import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi

def f(x):
    return 50 + 20 * np.sin(np.pi * x / 12) + 5 * np.exp(-0.2 * (x - 12)**2)

x = np.linspace(0, 24, 1000) 
y = f(x) 
 
plt.figure(figsize=(10, 6)) 
plt.plot(x, y, label=r'$f(x)=50+20\sin\left(\frac{\pi x}{12}\right)+5e^{-0.2(x-12)^2}$') 
plt.title('Графік функції навантаження на сервер') 
plt.xlabel('Час, x (год)') 
plt.ylabel('Навантаження, f(x)') 
plt.grid(True) 
plt.legend() 
plt.show()

a = 0
b = 24

I_0, _ = spi.quad(f, a, b)
print("Точне значення інтегралу I_0:", I_0)

def simpson(N):
    h = (b - a) / N
    sum_odd = 0
    sum_even = 0
    
    for i in range(1, N, 2):
        sum_odd += f(a + i * h)
        
    for i in range(2, N, 2):
        sum_even += f(a + i * h)
        
    return (h / 3) * (f(a) + 4 * sum_odd + 2 * sum_even + f(b))


N_values = []
eps_values = []
N_opt = None
eps_opt = None

for N in range(10, 1002, 2):
    I_N = simpson(N)
    eps = abs(I_N - I_0)
    
    N_values.append(N)
    eps_values.append(eps)
    
    if eps < 1e-12 and N_opt == None:
        N_opt = N
        eps_opt = eps

print("\nОптимальне число розбиттів N_opt:", N_opt)
print("Точність при N_opt:", eps_opt)

plt.figure(figsize=(8, 5))
plt.plot(N_values, eps_values)
plt.yscale('log') 
plt.xscale('log') 
plt.title('Залежність похибки від N')
plt.xlabel('Число вузлів N')
plt.ylabel('Похибка eps')
plt.grid()
plt.show()

print("\nБазовий Сімпсон")
N0_temp = int(N_opt / 10)
zalishok = N0_temp % 8

if zalishok != 0:
    N_0 = N0_temp + (8 - zalishok)
else:
    N_0 = N0_temp

I_N0 = simpson(N_0)
eps0 = abs(I_N0 - I_0)
print("Обране N_0:", N_0)
print("Базова похибка eps0:", eps0)

print("\nРунге-Ромберг")
I_N0_half = simpson(int(N_0 / 2))
I_R = I_N0 + (I_N0 - I_N0_half) / 15
epsR = abs(I_R - I_0)
print("Інтеграл Рунге-Ромберга I_R:", I_R)
print("Похибка Рунге-Ромберга epsR:", epsR)

print("\nЕйткен")
I_N0_quarter = simpson(int(N_0 / 4))
chiselnik_E = (I_N0_half ** 2) - (I_N0 * I_N0_quarter)
znamennik_E = 2 * I_N0_half - (I_N0 + I_N0_quarter)
I_E = chiselnik_E / znamennik_E
epsE = abs(I_E - I_0)

chiselnik_p = abs(I_N0_quarter - I_N0_half)
znamennik_p = abs(I_N0_half - I_N0)
p = (1 / np.log(2)) * np.log(chiselnik_p / znamennik_p)

print("Інтеграл методу Ейткена I_E:", I_E)
print("Оцінений порядок методу p:", p)
print("Похибка методу Ейткена epsE:", epsE)

print("\nПорівняння похибок різних методів")
print("1. Базовий Сімпсон:", eps0)
print("2. Рунге-Ромберг:  ", epsR)
print("3. Ейткен:         ", epsE)

methods = ['Базовий Сімпсон\n', 'Рунге-Ромберг\n', 'Ейткен\n']
errors = [eps0, epsR, epsE]

plt.figure(figsize=(8, 6))
plt.bar(methods, errors)
plt.yscale('log')
plt.title('Порівняння похибок різних методів для N_0 = ' + str(N_0))
plt.ylabel('Значення похибки')
plt.grid()
plt.show()

print("\nАдаптивний алгоритм")

calls_count = 0

def f_counted(x):
    global calls_count
    calls_count += 1
    return f(x)

def adaptive_simpson(a_val, b_val, delta):
    h = b_val - a_val
    mid = (a_val + b_val) / 2
    
    I1 = (h / 6) * (f_counted(a_val) + 4 * f_counted(mid) + f_counted(b_val))
    
    mid1 = (a_val + mid) / 2
    mid2 = (mid + b_val) / 2
    I2 = (h / 12) * (f_counted(a_val) + 4 * f_counted(mid1) + f_counted(mid)) + \
         (h / 12) * (f_counted(mid) + 4 * f_counted(mid2) + f_counted(b_val))
    
    if abs(I1 - I2) <= delta:
        return I2
    else:
        return adaptive_simpson(a_val, mid, delta) + adaptive_simpson(mid, b_val, delta)

deltas_test = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
actual_errors = []
calls_history = []

for d in deltas_test:
    calls_count = 0 
    
    I_adapt = adaptive_simpson(a, b, d)
    real_eps = abs(I_adapt - I_0)
    
    actual_errors.append(real_eps)
    calls_history.append(calls_count)
    
    print(f"При delta = {d}: Похибка = {real_eps:.2e}, Викликів = {calls_count}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(deltas_test, actual_errors, marker='o')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title(r'Залежність точності від заданого параметра $\delta$')
ax1.set_xlabel(r'Заданий параметр $\delta$')
ax1.set_ylabel('Реальна похибка обчислення')
ax1.grid()

ax2.plot(deltas_test, calls_history, marker='o')
ax2.set_xscale('log')
ax2.set_title(r'Залежність кількості обчислень від параметра $\delta$')
ax2.set_xlabel(r'Заданий параметр $\delta$')
ax2.set_ylabel('Кількість викликів функції')
ax2.grid()

plt.tight_layout()
plt.show()