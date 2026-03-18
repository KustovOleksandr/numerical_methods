import csv
import matplotlib.pyplot as plt

def form_matrix(x, m):
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(m + 1):
            A[i][j] = sum((xk ** (i + j)) for xk in x)
    return A

def form_vector(x, y, m):
    b = [0.0] * (m + 1)
    for i in range(m + 1):
        b[i] = sum(y[k] * (x[k] ** i) for k in range(len(x)))
    return b

def gauss_solve(A_in, b_in):
    n = len(A_in) #прямий хід
    A = [row[:] for row in A_in]
    b = b_in[:]
    for k in range(n - 1):
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[max_row][k]): max_row = i
        A[k], A[max_row] = A[max_row], A[k]
        b[k], b[max_row] = b[max_row], b[k]
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n): A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]
    x_sol = [0.0] * n # зворотній хід
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x_sol[j] for j in range(i + 1, n))
        x_sol[i] = (b[i] - sum_ax) / A[i][i]
    return x_sol

def polynomial(x_list, coef):
    y_poly = []
    for xi in x_list:
        val = sum(coef[i] * (xi ** i) for i in range(len(coef)))
        y_poly.append(val)
    return y_poly

def variance(y_true, y_approx):
    n = len(y_true)
    return sum((y_true[i] - y_approx[i]) ** 2 for i in range(n)) / n

def main():
    x = []
    y = []
    with open('temperature.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    max_degree = 10
    variances = []
    all_coefs = {}

    print("Дисперсії для різних ступенів m ")
    for m in range(1, max_degree + 1):
        A = form_matrix(x, m)
        b_vec = form_vector(x, y, m)
        coef = gauss_solve(A, b_vec)
        
        y_at_nodes = polynomial(x, coef)
        var = variance(y, y_at_nodes)
        
        variances.append(var)
        all_coefs[m] = coef
        print(f"Ступінь m={m:2}: дисперсія = {var:10.4f}") 

    optimal_m = variances.index(min(variances)) + 1
    optimal_coef = all_coefs[optimal_m]
    print(f"\nОптимальний ступінь: m = {optimal_m}") 

    x_future = [25, 26, 27]
    y_future = polynomial(x_future, optimal_coef)
    
    print("\nПрогноз температури на наступні 3 місяці")
    for i in range(len(x_future)):
        print(f"Місяць {int(x_future[i])}: {y_future[i]:.2f} °C") # 
    
    y_opt_nodes = polynomial(x, optimal_coef)
    errory = [abs(y[i] - y_opt_nodes[i]) for i in range(len(y))]

    plt.figure(figsize=(10, 12))
    plt.subplot(3, 1, 1)
    plt.plot(range(1, max_degree + 1), variances, 'ro-')
    plt.title("Дисперсія від степеня m")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.scatter(x, y, label='Дані')
    x_h = [x[0] + i*0.1 for i in range(int((x[-1]-x[0])*10)+1)]
    plt.plot(x_h, polynomial(x_h, optimal_coef), 'g-', label=f'm={optimal_m}')
    plt.scatter(x_future, y_future, color='purple', marker='*', s=100, label='Прогноз')
    plt.title("Апроксимація та Прогноз")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.bar(x, errory, color='orange', alpha=0.6)
    plt.title("Похибка у вузлах")
    plt.xticks(x)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()