import numpy as np

np.random.seed(67)
n = 100

A = np.random.uniform(1.0, 10.0, (n, n))
np.savetxt("matrix_A.txt", A)

X_2_5 = np.full(n, 2.5)

B = A @ X_2_5

np.savetxt("vector_B.txt", B)

def LU(mat_A):
    L = np.zeros((n,n))
    U = np.eye(n) 

    for k in range(n):
        for i in range(k, n):
            L[i, k] = mat_A[i, k] - np.dot(L[i, :k], U[:k, k])
            
        for j in range(k + 1, n):
            U[k, j] = (mat_A[k, j] - np.dot(L[k, :k], U[:k, j])) / L[k, k]
            
    return L, U

def solve_system(L, U, B_vec):
    Z = np.zeros(n)
    for k in range(n):
        Z[k] = (B_vec[k] - np.dot(L[k, :k], Z[:k])) / L[k, k]

    X = np.zeros(n)
    for k in range(n - 1, -1, -1):
        X[k] = Z[k] - np.dot(U[k, k+1:], X[k+1:])
        
    return X

A_read = np.loadtxt("matrix_A.txt")
B_read = np.loadtxt("vector_B.txt")

L_mat, U_mat = LU(A_read)

np.savetxt("matrix_L.txt", L_mat)
np.savetxt("matrix_U.txt", U_mat)

X0 = solve_system(L_mat, U_mat, B_read)

AX0 = A_read @ X0
eps = np.max(np.abs(AX0 - B_read))
print(f"Точність знайденого розв'язку: {eps}")

eps0 = 1e-12
iterations = 0

X_current = X0.copy()

while True:
    R = B_read - (A_read @ X_current)
    
    delta_X = solve_system(L_mat, U_mat, R)
    
    X_current = X_current + delta_X
    
    norm_delta_X = np.max(np.abs(delta_X))
    norm_R = np.max(np.abs(R))
    
    iterations += 1
    
    if norm_delta_X <= eps0 and norm_R <= eps0:
        print(f"\nДосягнуто заданої точності ({eps0})")
        print(f"Кількість ітерацій: {iterations}")
        print(f"Уточнений розв'язок СЛАР: {X_current}")
        break