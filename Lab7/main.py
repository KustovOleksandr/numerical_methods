import numpy as np

np.random.seed(42)
n = 100
A = np.random.uniform(1.0, 10.0, (n, n))

row_sums = np.sum(np.abs(A), axis=1)
np.fill_diagonal(A, row_sums)
np.savetxt("matrix_A.txt", A)

x_exact = np.full(n, 2.5)

B = A @ x_exact 
np.savetxt("vector_B.txt", B)

def simple_iteration_method(A, B, eps, max_iter):
    x = np.ones(100) 
    
    norm_A = np.max(np.sum(np.abs(A), axis=1)) 
    r = 2.0 / norm_A
    
    for k in range(max_iter):
        x_new = x - r * (A @ x - B)

        if np.max(np.abs(x_new - x)) < eps:
            return x_new, k + 1
            
        x = x_new
        
    return x, max_iter

def jacobi_method(A, B, eps, max_iter):
    x = np.ones(100)
    
    D = np.diag(A)
    R = A - np.diag(D)
    
    for k in range(max_iter):
        x_new = (B - R @ x) / D
        
        if np.max(np.abs(x_new - x)) < eps:
            return x_new, k + 1
            
        x = x_new
        
    return x, max_iter

def seidel_method(A, B, eps, max_iter):
    x = np.ones(100)
    
    for k in range(max_iter):
        x_old = np.copy(x) 
        
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i+1:], x_old[i+1:])
            
            x[i] = (B[i] - s1 - s2) / A[i, i]
            
        if np.max(np.abs(x - x_old)) < eps:
            return x, k + 1
            
    return x, max_iter

A_from_file = np.loadtxt("matrix_A.txt")
B_from_file = np.loadtxt("vector_B.txt")

eps_0 = 1e-14 
maximum_iterations = 10000

res_simple, iter_simple = simple_iteration_method(A_from_file, B_from_file, eps_0, maximum_iterations)
print("Метод простої ітерації:")
print("Кількість ітерацій:", iter_simple)
print("Перші 10 значення X:", res_simple[:10])

res_jacobi, iter_jacobi = jacobi_method(A_from_file, B_from_file, eps_0, maximum_iterations)
print("\nМетод Якобі:")
print("Кількість ітерацій:", iter_jacobi)
print("Перші 10 значення X:", res_jacobi[:10])

res_seidel, iter_seidel = seidel_method(A_from_file, B_from_file, eps_0, maximum_iterations)
print("\nМетод Зейделя:")
print("Кількість ітерацій:", iter_seidel)
print("Перші 10 значення X:", res_seidel[:10])