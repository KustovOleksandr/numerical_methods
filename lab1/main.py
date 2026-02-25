import requests
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
data = response.json()
results = data["results"]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

latitudes = [p["latitude"] for p in results] #широта
longitudes = [p["longitude"] for p in results] #довгота
elevations = [p["elevation"] for p in results] #висота
n = len(results)

distances = [0.0]
for i in range(1, n):
    d = haversine(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i]) # кумулятивна відстань
    distances.append(distances[-1] + d)

print("№ |   Висота    |  Дистанція")
for i in range(n):
    print(f"{i:2d}| {elevations[i]:8.2f}    | {distances[i]:10.2f} ")

def progonka(alpha, beta, gamma, delta):
    n = len(beta)
    A, B, x = [0.0]*n, [0.0]*n, [0.0]*n
    A[0], B[0] = -gamma[0]/beta[0], delta[0]/beta[0]
    for i in range(1, n-1):
        znam = alpha[i]*A[i-1] + beta[i]         # знаменник
        A[i] = -gamma[i]/znam                    # }
        B[i] = (delta[i] - alpha[i]*B[i-1])/znam # } допоміжні коеф в прогонці
    x[n-1] = (delta[n-1]-alpha[n-1]*B[n-2])/(alpha[n-1]*A[n-2]+beta[n-1]) # пряма прогонка
    for i in range(n-2, -1, -1):
        x[i] = A[i]*x[i+1] + B[i] # зворотня прогонка
    return x

h = [distances[i] - distances[i-1] for i in range(1, n)]
a_s, b_s, g_s, d_s = [0.0]*n, [1.0]*n, [0.0]*n, [0.0]*n
for i in range(1, n-1):
    a_s[i] = h[i-1]
    b_s[i] = 2 * (h[i-1] + h[i])
    g_s[i] = h[i]
    d_s[i] = 3 * ((elevations[i+1]-elevations[i])/h[i] - (elevations[i]-elevations[i-1])/h[i-1])

c_coeffs = progonka(a_s, b_s, g_s, d_s)
a_coeffs = elevations[:-1]
b_coeffs, d_coeffs = [], []

print("\n     КОЕФІЦІЄНТИ СПЛАЙНІВ")
for i in range(n-1):
    val_d = (c_coeffs[i+1] - c_coeffs[i]) / (3 * h[i])
    val_b = (elevations[i+1] - elevations[i])/h[i] - (h[i]*(c_coeffs[i+1] + 2*c_coeffs[i]))/3
    d_coeffs.append(val_d)
    b_coeffs.append(val_b)
    print(f"{i+1}: a={a_coeffs[i]:.2f}, b={val_b:.4f}, c={c_coeffs[i]:.4f}, d={val_d:.6f}")

print("\n     ХАРАКТЕРИСТИКИ МАРШРУТУ")
total_dist = distances[-1]
print(f"Загальна довжина маршруту: {total_dist:.2f} м") 

total_ascent = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n))
print(f"Сумарний набір висоти: {total_ascent:.2f} м") 

total_descent = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, n))
print(f"Сумарний спуск: {total_descent:.2f} м") 

gradients = [(elevations[i] - elevations[i-1])/h[i-1] * 100 for i in range(1, n)]
print(f"Максимальний підйом: {max(gradients):.2f} %") 
print(f"Максимальний спуск: {min(gradients):.2f} %") 
print(f"Середній градієнт: {np.mean(np.abs(gradients)):.2f} %") 

mass, g = 80, 9.81
energy_j = mass * g * total_ascent 
print(f"Механічна робота: {energy_j/1000:.2f} кДж") 
print(f"Енергія: {energy_j / 4184:.2f} ккал") # 

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Порівняння профілів при різній кількості вузлів', fontsize=16)

node_steps = [10, 15, 20, 21]
plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for count, pos in zip(node_steps, plot_positions):
    ax = axs[pos[0], pos[1]]
    
    d_sub = distances[:count]
    e_sub = elevations[:count]

    ax.scatter(d_sub, e_sub, color='red', s=30, label='Вузли')
    
    x_sub = np.linspace(min(d_sub), max(d_sub), 200)
    y_sub = []
    
    for x in x_sub:
        idx = 0
        for j in range(count - 1):
            if distances[j] <= x <= distances[j+1]:
                idx = j
                break
        dx = x - distances[idx]
        y_val = a_coeffs[idx] + b_coeffs[idx]*dx + c_coeffs[idx]*(dx**2) + d_coeffs[idx]*(dx**3) # рівняння кубічного сплайна
        y_sub.append(y_val)
        
    ax.plot(x_sub, y_sub, color='blue', label='Сплайн')
    ax.set_title(f'Кількість вузлів: {count}')
    ax.set_xlabel('Відстань (м)')
    ax.set_ylabel('Висота (м)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
