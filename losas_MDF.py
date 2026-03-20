import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. Parámetros geométricos y del material
Lx = 3.0        # Longitud en x (m)
Ly = 5.0        # Longitud en y (m)
h = 0.10        # Espesor de la losa (m)
E = 30.67e9     # Módulo de elasticidad del hormigón (Pa) -> 30 GPa
nu = 0.2        # Coeficiente de Poisson
q_load = 4000   # Carga transversal distribuida (N/m^2)

# Rigidez a la flexión de la placa (D)
D = (E * h**3) / (12 * (1 - nu**2))

# Condiciones de Borde [-1]--> simplemente apoyado; [+1]--> empotrado
izq = +1
der = +1
sup = -1
inf = -1

# 2. Discretización de la malla
delta = 0.2    # Tamaño del paso de la malla
Nx = int(Lx / delta)
Ny = int(Ly / delta)

# Solo resolvemos para los nodos INTERNOS
nx_in = Nx - 1
ny_in = Ny - 1
N_total = nx_in * ny_in 

print(f"La losa se dividió en {Nx}x{Ny} elementos.")
print(f"El sistema a resolver tendrá {N_total} nodos internos.")

# Función para mapear subíndices de la grilla 2D (i,j) al índice 1D (k)
def get_k(i, j):
    return i + j * nx_in

# Matriz dispersa del sistema (A) y vector de cargas (b)
A = sp.dok_matrix((N_total, N_total), dtype=float)
b = np.full(N_total, q_load / D)

factor = 1.0 / (delta**4)

for j in range(ny_in):
    for i in range(nx_in):
        k = get_k(i, j)
        
        # 1. Coeficiente central base
        coef_central = 20.0

        # Borde izquierdo
        if i == 0 and izq == -1:coef_central            -= 1.0  
        elif i == 0 and izq == +1:coef_central          += 1.0 

        # Borde derecho 
        if i == nx_in - 1 and der == -1: coef_central   -= 1.0  
        elif i == nx_in - 1 and der == +1: coef_central += 1.0

        # Borde inferior
        if j == 0 and inf == -1: coef_central           -= 1.0   
        elif j == 0 and inf == +1: coef_central         += 1.0

        # Borde superior
        if j == ny_in - 1 and sup == -1: coef_central   -= 1.0   
        elif j == ny_in - 1 and sup == +1: coef_central += 1.0  

        A[k, k] = coef_central * factor
        
        # 3. Nodos adyacentes a distancia 1 (en cruz)
        if i > 0: A[k, get_k(i-1, j)] += -8.0 * factor
        if i < nx_in-1: A[k, get_k(i+1, j)] += -8.0 * factor
        if j > 0: A[k, get_k(i, j-1)] += -8.0 * factor
        if j < ny_in-1: A[k, get_k(i, j+1)] += -8.0 * factor
        
        # 4. Nodos a distancia 2 (extremos de la cruz)
        if i > 1: A[k, get_k(i-2, j)] += 1.0 * factor
        if i < nx_in-2: A[k, get_k(i+2, j)] += 1.0 * factor
        if j > 1: A[k, get_k(i, j-2)] += 1.0 * factor
        if j < ny_in-2: A[k, get_k(i, j+2)] += 1.0 * factor
        
        # 5. Nodos diagonales
        if i > 0 and j > 0: A[k, get_k(i-1, j-1)] += 2.0 * factor
        if i > 0 and j < ny_in-1: A[k, get_k(i-1, j+1)] += 2.0 * factor
        if i < nx_in-1 and j > 0: A[k, get_k(i+1, j-1)] += 2.0 * factor
        if i < nx_in-1 and j < ny_in-1: A[k, get_k(i+1, j+1)] += 2.0 * factor

A = A.tocsr()
print("Matriz de rigidez ensamblada")

# ==========================================
# 4. RESOLUCIÓN DEL SISTEMA
# ==========================================
print("Resolviendo el sistema de ecuaciones...")
w_1D = spla.spsolve(A, b)

# ==========================================
# 5. RECONSTRUCCIÓN DE LA MALLA 2D
# ==========================================
W_2D = np.zeros((Ny + 1, Nx + 1))

for j in range(ny_in):
    for i in range(nx_in):
        W_2D[j+1, i+1] = w_1D[get_k(i, j)]

flecha_max = np.max(W_2D)
print(f"Resolución completa. Flecha máxima: {flecha_max * 1000:.4f} mm")

# ==========================================
# 6. CÁLCULO DE MOMENTOS FLECTORES
# ==========================================
d2w_dx2 = np.zeros_like(W_2D)
d2w_dy2 = np.zeros_like(W_2D)

# A. Curvatura en el interior de la losa
for j in range(1, ny_in + 1):      
    for i in range(1, nx_in + 1):  
        d2w_dx2[j, i] = (W_2D[j, i+1] - 2*W_2D[j, i] + W_2D[j, i-1]) / delta**2
        d2w_dy2[j, i] = (W_2D[j+1, i] - 2*W_2D[j, i] + W_2D[j-1, i]) / delta**2

# B. Curvatura en los bordes EMPOTRADOS
if sup == +1:
    for i in range(1, nx_in + 1):
        d2w_dy2[ny_in + 1, i] = (W_2D[ny_in, i] * 2) / delta**2
if inf == +1:
    for i in range(1, nx_in + 1):
        d2w_dy2[0, i] = (W_2D[1, i] * 2)/ delta**2
if izq == +1:
    for j in range(1, ny_in + 1):
        d2w_dx2[j, 0] = (W_2D[j, 1] * 2) / delta**2
if der == +1:
    for j in range(1, ny_in + 1):
        d2w_dx2[j, nx_in + 1] = (W_2D[j, nx_in] * 2) / delta**2

Mx = -D * (d2w_dx2 + nu * d2w_dy2)
My = -D * (d2w_dy2 + nu * d2w_dx2)

# Momento máximo para dimensionamiento
Mx_max = np.max(Mx)
Mx_min = np.min(Mx)
My_max = np.max(My)
My_min = np.min(My)
print(f"Momento en el tramo Mx: {Mx_max:.2f} Nm/m")
print(f"Momento en el apoyo Mx: {Mx_min:.2f} Nm/m")
print(f"Momento en el tramo My: {My_max:.2f} Nm/m")
print(f"Momento en el apoyo My: {My_min:.2f} Nm/m")

# ==========================================
# 7. VISUALIZACIÓN DE CONTORNOS (ISOLÍNEAS)
# ==========================================
x_vals = np.linspace(0, Lx, nx_in + 2)
y_vals = np.linspace(0, Ly, ny_in + 2)
X, Y = np.meshgrid(x_vals, y_vals)

Mx_kNm = Mx / 1000.0    # [kNm/m]
My_kNm = My / 1000.0    # [kNm/m]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico para Mx
contour1 = ax1.contourf(X, Y, Mx_kNm, levels=15, cmap='inferno')
ax1.set_title('Momentos Flectores $M_x$ (kNm/m)', fontsize=13)
ax1.set_xlabel('Eje X (m)')
ax1.set_ylabel('Eje Y (m)')
ax1.set_aspect('equal') # Mantener proporción de la losa en 2D
fig.colorbar(contour1, ax=ax1, label='Momento (kNm/m)')

# Gráfico para My
contour2 = ax2.contourf(X, Y, My_kNm, levels=15, cmap='inferno')
ax2.set_title('Momentos Flectores $M_y$ (kNm/m)', fontsize=13)
ax2.set_xlabel('Eje X (m)')
ax2.set_ylabel('Eje Y (m)')
ax2.set_aspect('equal')
fig.colorbar(contour2, ax=ax2, label='Momento (kNm/m)')

# Deformada
fig1 = plt.figure(figsize=(12, 8))
ax = fig1.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, -W_2D * 1000, cmap='plasma', edgecolor='k', linewidth=0.3, alpha=0.9)
ax.set_title('Deformada de Losa de Hormigón Simplemente Apoyada', fontsize=14, pad=20)
ax.set_box_aspect((Lx, Ly, 1.0))
fig1.colorbar(surf, shrink=0.5, aspect=10, label='Deformación (mm)')

plt.tight_layout()
plt.show()