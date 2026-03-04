import requests
import numpy as np
import matplotlib.pyplot as plt

locations = [
    {"latitude": 48.164214, "longitude": 24.536044},
    {"latitude": 48.164983, "longitude": 24.534836},
    {"latitude": 48.165605, "longitude": 24.534068},
    {"latitude": 48.166228, "longitude": 24.532915},
    {"latitude": 48.166777, "longitude": 24.531927},
    {"latitude": 48.167326, "longitude": 24.530884},
    {"latitude": 48.167011, "longitude": 24.530061},
    {"latitude": 48.166053, "longitude": 24.528039},
    {"latitude": 48.166655, "longitude": 24.526064},
    {"latitude": 48.166497, "longitude": 24.523574},
    {"latitude": 48.166128, "longitude": 24.520214},
    {"latitude": 48.165416, "longitude": 24.517170},
    {"latitude": 48.164546, "longitude": 24.514640},
    {"latitude": 48.163412, "longitude": 24.512980},
    {"latitude": 48.162331, "longitude": 24.511715},
    {"latitude": 48.162015, "longitude": 24.509462},
    {"latitude": 48.162147, "longitude": 24.506932},
    {"latitude": 48.161751, "longitude": 24.504244},
    {"latitude": 48.161197, "longitude": 24.501793},
    {"latitude": 48.160580, "longitude": 24.500537},
    {"latitude": 48.160250, "longitude": 24.500106},
]

url = "https://api.open-elevation.com/api/v1/lookup"
try:
    response = requests.post(url, json={"locations": locations}, timeout=15)
    response.raise_for_status()
    data = response.json()
    results = data["results"]
except (requests.RequestException, ValueError, KeyError) as e:
    print(f"Помилка при зверненні до API: {e}")
    print("Використовуємо збережені дані...")
    results = [
        {"latitude": 48.164214, "longitude": 24.536044, "elevation": 1264},
        {"latitude": 48.164983, "longitude": 24.534836, "elevation": 1285},
        {"latitude": 48.165605, "longitude": 24.534068, "elevation": 1285},
        {"latitude": 48.166228, "longitude": 24.532915, "elevation": 1333},
        {"latitude": 48.166777, "longitude": 24.531927, "elevation": 1310},
        {"latitude": 48.167326, "longitude": 24.530884, "elevation": 1318},
        {"latitude": 48.167011, "longitude": 24.530061, "elevation": 1318},
        {"latitude": 48.166053, "longitude": 24.528039, "elevation": 1339},
        {"latitude": 48.166655, "longitude": 24.526064, "elevation": 1375},
        {"latitude": 48.166497, "longitude": 24.523574, "elevation": 1417},
        {"latitude": 48.166128, "longitude": 24.520214, "elevation": 1486},
        {"latitude": 48.165416, "longitude": 24.517170, "elevation": 1524},
        {"latitude": 48.164546, "longitude": 24.514640, "elevation": 1553},
        {"latitude": 48.163412, "longitude": 24.512980, "elevation": 1630},
        {"latitude": 48.162331, "longitude": 24.511715, "elevation": 1757},
        {"latitude": 48.162015, "longitude": 24.509462, "elevation": 1794},
        {"latitude": 48.162147, "longitude": 24.506932, "elevation": 1828},
        {"latitude": 48.161751, "longitude": 24.504244, "elevation": 1887},
        {"latitude": 48.161197, "longitude": 24.501793, "elevation": 1975},
        {"latitude": 48.160580, "longitude": 24.500537, "elevation": 1975},
        {"latitude": 48.160250, "longitude": 24.500106, "elevation": 2031},
    ]
n = len(results)

latitudes = [p["latitude"] for p in results]
longitudes = [p["longitude"] for p in results]
elevations = [p["elevation"] for p in results]

print("            Табуляція вузлів:") 
print("No | Latitude  | Longitude | Elevation (m)") 
for i, point in enumerate(results): 
    print(f"{i:2d} | {point['latitude']:.6f} | " 
          f"{point['longitude']:.6f} | " 
          f"{point['elevation']:.2f}") 


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

distances = [0.0]
for i in range(1, n):
    d = haversine(latitudes[i-1], longitudes[i-1], latitudes[i], longitudes[i])
    distances.append(distances[-1] + d)

print("\n№  |   Висота    |  Дистанція")
for i in range(n):
    print(f"{i:2d} | {elevations[i]:8.2f}    | {distances[i]:10.2f} ")

def progonka(alpha, beta, gamma, delta):
    k = len(beta)
    A, B, x = [0.0]*k, [0.0]*k, [0.0]*k
    A[0], B[0] = -gamma[0]/beta[0], delta[0]/beta[0]
    for i in range(1, k-1):
        znam = alpha[i]*A[i-1] + beta[i]
        A[i] = -gamma[i]/znam
        B[i] = (delta[i] - alpha[i]*B[i-1])/znam
    x[k-1] = (delta[k-1]-alpha[k-1]*B[k-2])/(alpha[k-1]*A[k-2]+beta[k-1])
    for i in range(k-2, -1, -1):
        x[i] = A[i]*x[i+1] + B[i]
    return x

def calculate_spline_coeffs(x_arr, y_arr):
    k = len(x_arr)
    h_arr = [x_arr[i] - x_arr[i-1] for i in range(1, k)]
    a_s, b_s, g_s, d_s = [0.0]*k, [1.0]*k, [0.0]*k, [0.0]*k
    for i in range(1, k-1):
        a_s[i] = h_arr[i-1]
        b_s[i] = 2 * (h_arr[i-1] + h_arr[i])
        g_s[i] = h_arr[i]
        d_s[i] = 3 * ((y_arr[i+1]-y_arr[i])/h_arr[i] - (y_arr[i]-y_arr[i-1])/h_arr[i-1])

    c_coeffs = progonka(a_s, b_s, g_s, d_s)
    a_coeffs = y_arr[:-1]
    b_coeffs, d_coeffs = [], []

    for i in range(k-1):
        val_d = (c_coeffs[i+1] - c_coeffs[i]) / (3 * h_arr[i])
        val_b = (y_arr[i+1] - y_arr[i])/h_arr[i] - (h_arr[i]*(c_coeffs[i+1] + 2*c_coeffs[i]))/3
        d_coeffs.append(val_d)
        b_coeffs.append(val_b)
        
    return a_coeffs, b_coeffs, c_coeffs, d_coeffs

def calculate_spline(x, x_vals, a, b, c, d):
    if x <= x_vals[0]: idx = 0
    elif x >= x_vals[-1]: idx = len(x_vals) - 2
    else:
        idx = 0
        for j in range(len(x_vals) - 1):
            if x_vals[j] <= x <= x_vals[j+1]:
                idx = j
                break
    dx = x - x_vals[idx]
    return a[idx] + b[idx]*dx + c[idx]*(dx**2) + d[idx]*(dx**3)

print("\n    ХАРАКТЕРИСТИКИ МАРШРУТУ")
total_dist = distances[-1]
print(f"Загальна довжина маршруту: {total_dist:.2f} м") 
total_ascent = sum(max(elevations[i] - elevations[i-1], 0) for i in range(1, n))
print(f"Сумарний набір висоти: {total_ascent:.2f} м") 
total_descent = sum(max(elevations[i-1] - elevations[i], 0) for i in range(1, n))
print(f"Сумарний спуск: {total_descent:.2f} м") 
h_full = [distances[i] - distances[i-1] for i in range(1, n)]
gradients = [(elevations[i] - elevations[i-1])/h_full[i-1] * 100 for i in range(1, n)]
print(f"Максимальний підйом: {max(gradients):.2f} %") 
print(f"Максимальний спуск: {min(gradients):.2f} %") 
print(f"Середній градієнт: {np.mean(np.abs(gradients)):.2f} %") 
mass, g = 80, 9.81
energy_j = mass * g * total_ascent 
print(f"Механічна робота: {energy_j/1000:.2f} кДж") 
print(f"Енергія: {energy_j / 4184:.2f} ккал")

x_dense = np.linspace(distances[0], distances[-1], 1000)
a_base, b_base, c_base, d_base = calculate_spline_coeffs(distances, elevations)
y_base = [calculate_spline(x, distances, a_base, b_base, c_base, d_base) for x in x_dense]

node_steps = [10, 15, 20]
error_results = {}

for count in node_steps:
    idx_arr = np.linspace(0, n-1, count).astype(int)
    d_sub = [distances[i] for i in idx_arr]
    e_sub = [elevations[i] for i in idx_arr]
    
    a_s, b_s, c_s, d_s = calculate_spline_coeffs(d_sub, e_sub)
    y_approx = [calculate_spline(x, d_sub, a_s, b_s, c_s, d_s) for x in x_dense]
    
    errors = np.abs(np.array(y_base) - np.array(y_approx))
    error_results[count] = errors
    
    print(f"\n     {count} вузлів ")
    print(f"Максимальна похибка: {np.max(errors)}")
    print(f"Середня похибка: {np.mean(errors)}")


fig1, axs = plt.subplots(2, 2, figsize=(15, 10))
fig1.suptitle('Порівняння профілів при різній кількості вузлів', fontsize=16)
plot_nodes = [10, 15, 20, 21]
plot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for count, pos in zip(plot_nodes, plot_positions):
    ax = axs[pos[0], pos[1]]
    idx_arr = np.linspace(0, n-1, count).astype(int)
    d_sub = [distances[i] for i in idx_arr]
    e_sub = [elevations[i] for i in idx_arr]
    
    a_s, b_s, c_s, d_s = calculate_spline_coeffs(d_sub, e_sub)
    print(f"\nКоефіцієнти кубічного сплайна для {count} вузлів:")
    print(" i |      a      |      b      |      c      |      d      ")
    print("-" * 65)
    for i in range(len(a_s)):
        print(f"{i:2d} | {a_s[i]:11.4f} | {b_s[i]:11.4f} | {c_s[i]:11.4f} | {d_s[i]:11.8f}")
    
    ax.scatter(d_sub, e_sub, color='red', s=30, label='Вузли')
    x_sub = np.linspace(min(d_sub), max(d_sub), 200)
    y_sub = [calculate_spline(x, d_sub, a_s, b_s, c_s, d_s) for x in x_sub]
        
    ax.plot(x_sub, y_sub, color='blue', label='Сплайн')
    ax.set_title(f'Кількість вузлів: {count}')
    ax.set_xlabel('Відстань (м)')
    ax.set_ylabel('Висота (м)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


fig2 = plt.figure(figsize=(8, 6))
plt.title('Похибка апроксимації')
for count in node_steps:
    plt.plot(x_dense, error_results[count], label=f'{count} вузлів')
plt.legend()

plt.show()