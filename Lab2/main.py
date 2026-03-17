import csv
import matplotlib.pyplot as plt
import numpy as np
import math

def read_data(filename):
    x_list = []
    y_list = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x_list.append(float(row['n']))
            y_list.append(float(row['t']))
    return x_list, y_list

def table_differences(x, y):
    n = len(y)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]
        
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])
    return table

def print_diff_table(x, table):
    print("\n ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ ")
    n = len(x)
    for i in range(n):
        row_str = f"x={x[i]:<6} | "
        for j in range(n - i):
            row_str += f"{table[i][j]:>12.12f} "
        print(row_str)

def newton_method(x_val, x_data, diff_table):
    n = len(x_data)
    result = diff_table[0][0]
    for j in range(1, n):
        omega = 1.0
        for i in range(j):
            omega *= (x_val - x_data[i])
        result += diff_table[0][j] * omega
    return result

def factorial_method(x_val, x_h, y_h):
    h = x_h[1] - x_h[0]
    t = (x_val - x_h[0]) / h 
    n = len(x_eq)
    
    diff_table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        diff_table[i][0] = y_h[i]
    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]
            
    result = diff_table[0][0]
    for k in range(1, n):
        t_k = 1.0
        for m in range(k):
            t_k *= (t - m)
        result += (diff_table[0][k] / math.factorial(k)) * t_k
    return result

def lagrange_method(x_val, x_data, y_data):
    n = len(x_data)
    result = 0.0
    for i in range(n):
        term = y_data[i]    
        for j in range(n):
            if i != j:
                term *= (x_val - x_data[j]) / (x_data[i] - x_data[j])
        result += term
    return result

def plot_main_graph(x_data, y_data, diff_table):
    plot_x = np.linspace(min(x_data), max(x_data), 100)
    plot_y = [newton_method(x, x_data, diff_table) for x in plot_x]
        
    plt.figure(figsize=(10, 6))
    plt.plot(plot_x, plot_y, color='blue', label='Многочлен Ньютона')
    plt.scatter(x_data, y_data, color='red', s=50, zorder=5, label='Експериментальні дані')
    plt.axhline(60, color='green', linestyle='--', label='Межа комфортної гри (60 FPS)')
    
    plt.title("Залежність FPS від кількості об'єктів")
    plt.xlabel("Кількість об'єктів (n)")
    plt.ylabel("Кількість кадрів за секунду (FPS)")
    plt.grid(True)
    plt.legend()
    plt.show()

def fixed_interval(x_data, diff_table):
    a, b = x_data[0], x_data[-1] # фіксований інтервал
    x_interval = np.linspace(a, b, 200)
    y_etalon = [newton_method(x, x_data, diff_table) for x in x_interval] 
    
    nodes_counts = [5, 10, 20]
    plt.figure(figsize=(12, 8))
    
    for n in nodes_counts:
        x_nodes = np.linspace(a, b, n)
        y_nodes = [newton_method(x, x_data, diff_table) for x in x_nodes]
        current_diff = table_differences(x_nodes, y_nodes)
        y_error = [newton_method(x, x_nodes, current_diff) for x in x_interval]
        
        errors = [abs(y_etalon[i] - y_error[i]) for i in range(len(x_interval))]
        plt.plot(x_interval, errors, label=f'Похибка (n={n})')

    plt.title("Фіксований інтервал")
    plt.xlabel("Кількість об'єктів")
    plt.ylabel("Абсолютна похибка")
    plt.legend()
    plt.grid(True)
    plt.show()

def fixed_step(x_data, diff_table):
    a = x_data[0]
    h = 40 # фіксований крок
    nodes_counts = [5, 10, 20]
    
    plt.figure(figsize=(12, 8))
    
    for n in nodes_counts:
        b = a + h * n 
        x_nodes = np.linspace(a, b, n)
        y_nodes = [newton_method(x, x_data, diff_table) for x in x_nodes]
        current_diff = table_differences(x_nodes, y_nodes)
        
        x_interval = np.linspace(a, b, 200)
        y_etalon = [newton_method(x, x_data, diff_table) for x in x_interval]
        y_error = [newton_method(x, x_nodes, current_diff) for x in x_interval]
        
        errors = [abs(y_etalon[i] - y_error[i]) for i in range(len(x_interval))]
        plt.plot(x_interval, errors, label=f'n={n}, відрізок [{a}, {b}]')

    plt.title(f"Фіксований крок (h={h})")
    plt.xlabel("Кількість об'єктів")
    plt.ylabel("Абсолютна похибка")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    
    x_points, y_points = read_data("data.csv")
    table = table_differences(x_points, y_points)
    print_diff_table(x_points, table)
    
    fps_newton = newton_method(1000, x_points, table)
    
    x_eq = np.linspace(x_points[0], x_points[-1], 5)
    y_eq = [newton_method(x, x_points, table) for x in x_eq]
    fps_factorial = factorial_method(1000, x_eq, y_eq)
    
    fps_lagrange = lagrange_method(1000, x_points, y_points)
    
    print("\n ПРОГНОЗ ДЛЯ 1000 ОБ'ЄКТІВ ")
    print(f"Метод Ньютона:           {round(fps_newton, 4)} FPS")
    print(f"Факторіальний метод: {round(fps_factorial, 4)} FPS")
    print(f"Метод Лагранжа:          {round(fps_lagrange, 4)} FPS")
    
    plot_main_graph(x_points, y_points, table)
    fixed_interval(x_points, table)  
    fixed_step(x_points, table)     
    