import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 - y

def analit_result(x):
    return x**2 - 2*x + 2 - np.exp(-x)


x0 = 0.0
y0 = 1.0
x_end = 2.0
h_fixed = 0.01 
tol = 1e-3     

def rung_cut_4(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h * k1 / 2)
    k3 = f(x + h/2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)


def adams_fixed_step(f, x0, y0, x_end, h):
    n_steps = int((x_end - x0) / h)
    x = np.linspace(x0, x_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    y[1] = rung_cut_4(f, x[0], y[0], h)
    
    pre_errors = []
    true_errors = [0, y[1] - analit_result(x[1])]
    
    for i in range(1, n_steps):
        fn = f(x[i], y[i])
        fn_minus_1 = f(x[i-1], y[i-1])
        
        y_pre = y[i] + (h / 2) * (3 * fn - fn_minus_1)
        
        fn_plus_1_pre = f(x[i+1], y_pre)
        y_cor = y[i] + (h / 2) * (fn_plus_1_pre + fn)
        
        y[i+1] = y_cor
        
        pre_errors.append(abs(y_cor - y_pre))
        true_errors.append(y_cor - analit_result(x[i+1]))
        
    return x, y, np.array(true_errors), np.pad(pre_errors, (2, 0), mode='constant')

def adams_adaptive_step(f, x0, y0, x_end, h0, tol):
    x_vals, y_vals, h_vals = [x0], [y0], [h0]
    h = h0
    x, y = x0, y0
    
    y = rung_cut_4(f, x, y, h)
    x += h
    x_vals.append(x); y_vals.append(y); h_vals.append(h)
    
    while x < x_end:
        fn = f(x_vals[-1], y_vals[-1])
        fn_minus_1 = f(x_vals[-2], y_vals[-2])
        
        y_pre = y_vals[-1] + (h / 2) * (3 * fn - fn_minus_1)
        y_cor = y_vals[-1] + (h / 2) * (f(x + h, y_pre) + fn)
        
        error = (1/6) * abs(y_cor - y_pre)
        
        if error > tol:
            h /= 2
            x = x_vals[-2] + h
            y = rung_cut_4(f, x_vals[-2], y_vals[-2], h)
            x_vals[-1] = x; y_vals[-1] = y; h_vals[-1] = h
            continue
            
        x += h
        x_vals.append(x); y_vals.append(y_cor)
        
        if error < tol / 4:
            h *= 2
            y_rk = rung_cut_4(f, x, y_cor, h)
            x += h
            x_vals.append(x); y_vals.append(y_rk); h_vals.extend([h, h])
        else:
            h_vals.append(h)
            
    return np.array(x_vals), np.array(h_vals[:len(x_vals)])


def rk4_fixed_step(f, x0, y0, x_end, h):
    n_steps = int((x_end - x0) / h)
    x = np.linspace(x0, x_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    true_errors = [0]
    runge_errors = [0]
    
    for i in range(n_steps):
        y_h = rung_cut_4(f, x[i], y[i], h)
        y[i+1] = y_h
        
        y_h2_1 = rung_cut_4(f, x[i], y[i], h/2)
        y_h2_2 = rung_cut_4(f, x[i] + h/2, y_h2_1, h/2)
        
        runge_err = (16/15) * abs(y_h - y_h2_2)
        runge_errors.append(runge_err)
        
        true_errors.append(y[i+1] - analit_result(x[i+1]))
        
    return x, y, np.array(true_errors), np.array(runge_errors)

def rk4_adaptive_step(f, x0, y0, x_end, h0, tol):
    x_vals, h_vals = [x0], [h0]
    x, y, h = x0, y0, h0
    
    while x < x_end:
        y_h = rung_cut_4(f, x, y, h)
        y_h2 = rung_cut_4(f, x + h/2, rung_cut_4(f, x, y, h/2), h/2)
        
        error = (16/15) * abs(y_h - y_h2)
        
        if error > tol:
            h /= 2
            continue 
            
        x += h
        y = y_h2 
        x_vals.append(x)
        
        if error < tol / 32:
            h *= 2
            
        h_vals.append(h)
        
    return np.array(x_vals), np.array(h_vals[:len(x_vals)])


x_adams, y_adams, err_true_adams, err_est_adams = adams_fixed_step(f, x0, y0, x_end, h_fixed)
x_rk4, y_rk4, err_true_rk4, err_runge_rk4 = rung_cut_4_with_errors = rk4_fixed_step(f, x0, y0, x_end, h_fixed)

x_adams_ad, h_adams_ad = adams_adaptive_step(f, x0, y0, x_end, h_fixed, tol)
x_rk4_ad, h_rk4_ad = rk4_adaptive_step(f, x0, y0, x_end, h_fixed, tol)

fig, axs = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)

axs[0, 0].plot(x_adams, err_true_adams, 'b.-', label=r"Істинна похибка $\phi_n$")
axs[0, 0].set_title("Метод Адамса: Локальна похибка")
axs[0, 0].grid(True); axs[0, 0].legend()

axs[0, 1].plot(x_adams, err_est_adams, 'r.-', label="$|y^{cor} - y^{pre}|$")
axs[0, 1].set_title("Метод Адамса: Оцінка похибки")
axs[0, 1].grid(True); axs[0, 1].legend()

axs[1, 0].step(x_adams_ad, h_adams_ad, 'g.-', where='post', label="Крок $h(x)$")
axs[1, 0].set_title(f"Метод Адамса: Адаптивний крок")
axs[1, 0].grid(True); axs[1, 0].legend()

axs[1, 1].plot(x_rk4, err_true_rk4, 'b.-', label=r"Істинна похибка $\phi_n$")
axs[1, 1].set_title("Метод Рунге-Кутта: Локальна похибка")
axs[1, 1].grid(True); axs[1, 1].legend()

axs[2, 0].plot(x_rk4, err_runge_rk4, 'm.-', label="Похибка Рунге")
axs[2, 0].set_title("Метод Рунге-Кутта: Оцінка похибки")
axs[2, 0].grid(True); axs[2, 0].legend()

axs[2, 1].step(x_rk4_ad, h_rk4_ad, 'g.-', where='post', label="Крок $h(x)$")
axs[2, 1].set_title(f"Метод Рунге-Кутта: Адаптивний крок")
axs[2, 1].grid(True); axs[2, 1].legend()


plt.show()