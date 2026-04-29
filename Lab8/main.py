import numpy as np
import matplotlib.pyplot as plt

def F(x):
    return np.sin(x) - 0.5 * x

def dF(x):
    return np.cos(x) - 0.5

def d2F(x):
    return -np.sin(x)

def tabulate(a, b, h, filename="tabulation.txt"):
    x_vals = np.arange(a, b + h/2, h)
    y_vals = F(x_vals)
    
    data = np.column_stack((x_vals, y_vals))
    np.savetxt(filename, data, fmt="%.4f", delimiter="\t\t", 
               header="x\t\tF(x)", comments="")
    
    indices = np.where(np.diff(np.sign(y_vals)))[0]
    
    approx_roots = []
    found_inc = False
    found_dec = False
    
    for idx in indices:
        x_prev, x_curr = x_vals[idx], x_vals[idx+1]
        y_prev, y_curr = y_vals[idx], y_vals[idx+1]
        
        behavior = "Зростає" if y_curr > y_prev else "Спадає"
        
        if behavior == "Зростає" and not found_inc:
            approx_roots.append({"x0": (x_prev + x_curr) / 2, "behavior": behavior})
            found_inc = True
        elif behavior == "Спадає" and not found_dec:
            approx_roots.append({"x0": (x_prev + x_curr) / 2, "behavior": behavior})
            found_dec = True
            
        if found_inc and found_dec: break
            
    return approx_roots

def simple_iteration(x0, tau, eps=1e-10):
    xn, iters = x0, 0
    while True:
        iters += 1
        x_next = xn + tau * F(xn)
        if abs(F(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, iters
        xn = x_next

def newton_method(x0, eps=1e-10):
    xn, iters = x0, 0
    while True:
        iters += 1
        x_next = xn - F(xn) / dF(xn)
        if abs(F(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, iters
        xn = x_next

def chebyshev_method(x0, eps=1e-10):
    xn, iters = x0, 0
    while True:
        iters += 1
        fx, dfx, d2fx = F(xn), dF(xn), d2F(xn)
        x_next = xn - fx/dfx - 0.5 * (fx**2 * d2fx) / (dfx**3)
        if abs(F(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, iters
        xn = x_next

def hord_method(x0, x1, eps=1e-10):
    xn_1, xn, iters = x0, x1, 0
    while True:
        iters += 1
        fn, fn_1 = F(xn), F(xn_1)
        x_next = xn - fn * (xn - xn_1) / (fn - fn_1)
        if abs(F(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, iters
        xn_1, xn = xn, x_next

def parabola_method(x0, x1, x2, eps=1e-10):
    xn_2, xn_1, xn = x0, x1, x2
    iters = 0
    
    while True:
        iters += 1
        
        F_n_n1 = (F(xn) - F(xn_1)) / (xn - xn_1)
        F_n_n1_n2 = (((F(xn) - F(xn_1)) / (xn - xn_1)) - ((F(xn_1) - F(xn_2)) / (xn_1 - xn_2))) / (xn - xn_2)
        
        under_sqrt = ((xn - xn_1) * F_n_n1_n2 + F_n_n1)**2 - 4 * F_n_n1_n2 * F(xn)
        root = np.lib.scimath.sqrt(under_sqrt)
        
        delta_plus = (1 / (2 * F_n_n1_n2)) * (-((xn - xn_1) * F_n_n1_n2 + F_n_n1) + root)
        delta_minus = (1 / (2 * F_n_n1_n2)) * (-((xn - xn_1) * F_n_n1_n2 + F_n_n1) - root)
        
        delta = delta_plus if abs(delta_plus) < abs(delta_minus) else delta_minus
        
        x_next = xn + delta.real
        
        if abs(F(x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, iters
        
        xn_2, xn_1, xn = xn_1, xn, x_next

def inverse_interpolation(nodes, eps=1e-10):
    x = list(nodes)
    iters = 0
    while True:
        iters += 1
        y = [F(val) for val in x]
        x_next = 0
        for i in range(len(x)):
            li = 1
            for j in range(len(x)):
                if i != j:
                    li *= (0 - y[j]) / (y[i] - y[j])
            x_next += x[i] * li
            
        if abs(F(x_next)) < eps and abs(x_next - x[-1]) < eps:
            return x_next, iters
        x = x[1:] + [x_next]

def coefs(filename="poly_coeffs.txt"):
    coeffs = np.array([-10.0, 9.0, -4.0, 1.0])
    np.savetxt(filename, coeffs)
    return np.loadtxt(filename)

def eval_poly_horner(A, x):
    return np.polyval(A[::-1], x)

def newton_horner_method(A, x0, eps=1e-10):
    xn, iters = x0, 0
    m = len(A) - 1
    while True:
        iters += 1
        b = np.zeros(m + 1)
        b[m] = A[m]
        for i in range(m - 1, -1, -1):
            b[i] = A[i] + xn * b[i + 1]
            
        c = np.zeros(m + 1)
        c[m] = b[m]
        for i in range(m - 1, 0, -1):
            c[i] = b[i] + xn * c[i + 1]
            
        x_next = xn - b[0] / c[1]
        if abs(eval_poly_horner(A, x_next)) < eps and abs(x_next - xn) < eps:
            return x_next, iters
        xn = x_next

def lin_method(A, alpha0, beta0, eps=1e-10):
    iters, alpha, beta = 0, alpha0, beta0
    m = len(A) - 1
    while True:
        iters += 1
        p, q = -2 * alpha, alpha**2 + beta**2
        b = np.zeros(m + 1)
        b[m] = A[m]
        b[m-1] = A[m-1] - p * b[m]
        for i in range(m - 2, 1, -1):
            b[i] = A[i] - p * b[i+1] - q * b[i+2]
            
        q1 = A[0] / b[2]
        p1 = (A[1] * b[2] - A[0] * (b[3] if m >= 3 else 0)) / (b[2]**2)
        
        alpha1 = -p1 / 2
        beta1 = np.lib.scimath.sqrt(q1 - alpha1**2).real
        
        if abs(alpha1 - alpha) <= eps and abs(beta1 - beta) <= eps:
            return complex(alpha1, beta1), complex(alpha1, -beta1), iters
        alpha, beta = alpha1, beta1

def plot(A):
    x_vals = np.linspace(-2, 5, 400)
    
    y_vals = np.polyval(A[::-1], x_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label='$F(x) = x^3 - 4x^2 + 9x - 10$', color='blue', linewidth=2)
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.scatter([2], [0], color='red', s=50, zorder=5, label='Дійсний корінь ($x=2$)')
    
    plt.title("Графік алгебраїчного рівняння 3-го порядку")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    approx_roots = tabulate(-1.0, 3.0, 0.1)
    
    eps = 1e-10
    for i, r in enumerate(approx_roots):
        x0, beh = r["x0"], r["behavior"]
        print(f"\nКорінь {i+1} (x0 ≈ {x0:.4f}, {beh}):")
        
        tau = -1 if beh == "Зростає" else 1
        print(f"  Проста ітерація: {simple_iteration(x0, tau, eps)}")
        print(f"  Метод Ньютона:   {newton_method(x0, eps)}")
        print(f"  Метод Чебишева:  {chebyshev_method(x0, eps)}")
        print(f"  Метод хорд:      {hord_method(x0-0.1, x0+0.1, eps)}")
        print(f"  Метод парабол:   {parabola_method(x0-0.1, x0, x0+0.1, eps)}")
        print(f"  Звор. інтерпол.: {inverse_interpolation([x0-0.1, x0, x0+0.1], eps)}")

    A = coefs()
    
    plot(A)
    
    res_real, it_r = newton_horner_method(A, 1.5, eps)
    print(f"Дійсний корінь (Горнер): {res_real:.10f} (Ітерацій: {it_r})")
    
    c1, c2, it_c = lin_method(A, 0.5, 1.5, eps)
    print(f"Комплексні корені (Лін): {c1}, {c2} (Ітерацій: {it_c})")