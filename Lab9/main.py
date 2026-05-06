import numpy as np
import matplotlib.pyplot as plt

def function(X):
    x1, x2 = X[0], X[1]
    return 100 * (x1**2 - x2)**2 + (x1 - 1)**2

def investigate_search(X_base, delta_X, q, eps1, func, decrease_step=True):
    X_new = X_base.copy()
    current_delta = delta_X.copy()
    
    for i in range(len(X_base)):
        while True:
            X_forward = X_new.copy()
            X_forward[i] += current_delta[i]
            if func(X_forward) < func(X_new):
                X_new = X_forward
                break 
            
            X_backward = X_new.copy()
            X_backward[i] -= current_delta[i]
            if func(X_backward) < func(X_new):
                X_new = X_backward
                break  
            
            if not decrease_step:
                break
            
            current_delta[i] /= q
            
            if current_delta[i] < eps1:
                break
                
    return X_new, current_delta

def hooke_jeeves(X0, delta_X0, q, p, eps1, eps2, func):
    X0 = np.array(X0, dtype=float)
    delta_X = np.array(delta_X0, dtype=float)
    
    X_base = X0.copy()
    path = [X_base.copy()]
    
    while True:
        X1, delta_X = investigate_search(X_base, delta_X, q, eps1, func, decrease_step=True)
        
        if np.array_equal(X1, X_base):
            break
            
        norm_delta = np.linalg.norm(delta_X)
        diff_func = abs(func(X1) - func(X_base))
        
        if norm_delta < eps1 and diff_func < eps2:
            X_base = X1
            path.append(X_base.copy())
            break
            
        while True:
            X2_p = X1 + p * (X1 - X_base)
            
            X2, _ = investigate_search(X2_p, delta_X, q, eps1, func, decrease_step=False)
            
            if func(X2) < func(X1):
                X_base = X1
                X1 = X2
                path.append(X1.copy())
            else:
                X_base = X1
                path.append(X_base.copy())
                break
                
    return X_base, path

def plot_graphic():
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-1, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)

    F1 = 10 * (X1**2 - X2)
    F2 = X1 - 1

    plt.figure(figsize=(8, 8))

    plt.contour(X1, X2, F1, levels=[0], colors='blue', linewidths=2)
    plt.contour(X1, X2, F2, levels=[0], colors='red', linewidths=2)
    
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    plt.plot([], [], color='blue', linewidth=2, label='$10(x_1^2 - x_2) = 0$')
    plt.plot([], [], color='red', linewidth=2, label='$x_1 - 1 = 0$')

    plt.title('Графіки рівнянь для функції Розенброка')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    X_start = [-1.2, 0.0] 
    
    delta_start = [0.5, 0.5] 
    
    q_param = 2.0        
    p_param = 2.0        
    epsilon_1 = 1e-8  
    epsilon_2 = 1e-8     
    
    print(f"Початкова точка: {X_start}")
    
    X_opt, trajectory = hooke_jeeves(X_start, delta_start, q_param, p_param, epsilon_1, epsilon_2, function)
    
    print(f"\nЗнайдений мінімум (X*): {X_opt}")
    print(f"Значення цільової функції в точці мінімуму: {function(X_opt):.6f}")
    print(f"Кількість кроків : {len(trajectory)}")
    
    filename = "trajectory.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Крок\tX1\t\tX2\t\tPhi(X)\n")
        f.write("-" * 50 + "\n")
        for step, point in enumerate(trajectory):
            f_val = function(point)
            f.write(f"{step}\t{point[0]:.6f}\t{point[1]:.6f}\t{f_val:.6f}\n")
            
    plot_graphic()