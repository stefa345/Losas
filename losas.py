import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. Parámetros geométricos y del material
Lx = 3.0        # Longitud en x (m)
Ly = 5.0        # Longitud en y (m)
h = 0.10        # Espesor de la losa (m)
E = 30.67e9        # Módulo de elasticidad del hormigón (Pa) -> 30 GPa
nu = 0.2        # Coeficiente de Poisson
q_load = 4000   # Carga transversal distribuida (N/m^2)

# Rigidez a la flexión de la placa (D)
D = (E * h**3) / (12 * (1 - nu**2))

# 2. Discretización de la malla
delta = 0.2    # Tamaño del paso de la malla dx = dy = delta (m)
Nx = int(Lx / delta)
Ny = int(Ly / delta)

# Solo resolvemos para los nodos INTERNOS (los bordes ya sabemos que w=0)
nx_in = Nx - 1
ny_in = Ny - 1
N_total = nx_in * ny_in  # Cantidad total de incógnitas

print(f"La losa se dividió en {Nx}x{Ny} elementos.")
print(f"El sistema a resolver tendrá {N_total} ecuaciones (nodos internos).")

# Función para mapear subíndices de la grilla 2D (i,j) al índice 1D (k)
def get_k(i, j):
    # i varía en X (0 a nx_in-1), j varía en Y (0 a ny_in-1)
    return i + j * nx_in

# Matriz dispersa del sistema (A) y vector de cargas (b)
A = sp.dok_matrix((N_total, N_total), dtype=float)
b = np.full(N_total, q_load / D)

factor = 1.0 / (delta**4)

izq = +1
der = +1
sup = +1
inf = +1

for j in range(ny_in):
    for i in range(nx_in):
        k = get_k(i, j)
        
        # 1. Coeficiente central base
        coef_central = 20.0
        
        # 2. Aplicamos la lógica de los nodos fantasmas para modificar el coeficiente central
        # Si estamos pegados a un borde (i=0, i=nx_in-1, j=0, j=ny_in-1), 
        # el nodo a distancia 2 cae afuera y se resta 1 al central.
        if i == 0 and izq == -1:coef_central            -= 1.0  # Borde izquierdo
        elif i == 0 and izq == +1:coef_central          += 1.0  # Borde izquierdo

        if i == nx_in - 1 and der == -1: coef_central   -= 1.0  # Borde derecho 
        elif i == nx_in - 1 and der == +1: coef_central += 1.0  # Borde derecho 

        if j == 0 and inf == -1: coef_central           -= 1.0  # Borde inferior 
        elif j == 0 and inf == +1: coef_central         += 1.0  # Borde inferior 

        if j == ny_in - 1 and sup == -1: coef_central   -= 1.0  # Borde superior 
        elif j == ny_in - 1 and sup == +1: coef_central += 1.0  # Borde superior 

        A[k, k] = coef_central * factor
        
        # 3. Nodos adyacentes a distancia 1 (cruz: arriba, abajo, izq, der)
        if i > 0: A[k, get_k(i-1, j)] += -8.0 * factor
        if i < nx_in-1: A[k, get_k(i+1, j)] += -8.0 * factor
        if j > 0: A[k, get_k(i, j-1)] += -8.0 * factor
        if j < ny_in-1: A[k, get_k(i, j+1)] += -8.0 * factor
        
        # 4. Nodos a distancia 2 (extremos de la cruz)
        # Solo los sumamos si no caen afuera (si caen afuera, ya los restamos del central arriba)
        if i > 1: A[k, get_k(i-2, j)] += 1.0 * factor
        if i < nx_in-2: A[k, get_k(i+2, j)] += 1.0 * factor
        if j > 1: A[k, get_k(i, j-2)] += 1.0 * factor
        if j < ny_in-2: A[k, get_k(i, j+2)] += 1.0 * factor
        
        # 5. Nodos diagonales
        if i > 0 and j > 0: A[k, get_k(i-1, j-1)] += 2.0 * factor
        if i > 0 and j < ny_in-1: A[k, get_k(i-1, j+1)] += 2.0 * factor
        if i < nx_in-1 and j > 0: A[k, get_k(i+1, j-1)] += 2.0 * factor
        if i < nx_in-1 and j < ny_in-1: A[k, get_k(i+1, j+1)] += 2.0 * factor

# Convertimos a un formato más rápido para resolver
A = A.tocsr()
print("¡Matriz de rigidez ensamblada con éxito!")

# ==========================================
# 4. RESOLUCIÓN DEL SISTEMA
# ==========================================
# Resolvemos el sistema lineal A * w = b
print("Resolviendo el sistema de ecuaciones...")
w_1D = spla.spsolve(A, b)

# ==========================================
# 5. RECONSTRUCCIÓN DE LA MALLA 2D
# ==========================================
# Creamos una matriz llena de ceros que incluya los bordes
W_2D = np.zeros((Ny + 1, Nx + 1))

# Llenamos el interior de la matriz con la solución que acabamos de hallar
for j in range(ny_in):
    for i in range(nx_in):
        W_2D[j+1, i+1] = w_1D[get_k(i, j)]

# Extraemos la deformación máxima (en el centro de la losa)
flecha_max = np.max(W_2D)
print(f"Resolución completa. Flecha máxima en el centro: {flecha_max * 1000:.4f} mm")

# ==========================================
# 6. CÁLCULO DE MOMENTOS FLECTORES
# ==========================================
# Inicializamos matrices de ceros para las derivadas y los momentos
# El tamaño es el mismo que la losa completa, así los bordes quedan en 0 automáticamente
d2w_dx2 = np.zeros_like(W_2D)
d2w_dy2 = np.zeros_like(W_2D)

# A. Curvatura en el interior de la losa
for j in range(1, ny_in + 1):      # Filas (eje Y)
    for i in range(1, nx_in + 1):  # Columnas (eje X)
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

    
# Aplicamos las ecuaciones de placa elástica
Mx = -D * (d2w_dx2 + nu * d2w_dy2)
My = -D * (d2w_dy2 + nu * d2w_dx2)

# Momento máximo para dimensionamiento
Mx_max = np.max(Mx)
Mx_min = np.min(Mx)
My_max = np.max(My)
My_min = np.min(My)
print(f"Momento en el tramo Mx: {Mx_max:.4f} Nm/m")
print(f"Momento en el apoyo Mx: {Mx_min:.4f} Nm/m")
print(f"Momento en el tramo My: {My_max:.4f} Nm/m")
print(f"Momento en el tramo My: {My_min:.4f} Nm/m")

# ==========================================
# 7. VISUALIZACIÓN DE CONTORNOS (ISOLÍNEAS)
# ==========================================
# Creamos las coordenadas para la malla del gráfico
x_vals = np.linspace(0, Lx, nx_in + 2)
y_vals = np.linspace(0, Ly, ny_in + 2)
X, Y = np.meshgrid(x_vals, y_vals)

# Pasamos los momentos a kNm/m dividiendo por 1000 para mejor lectura
Mx_kNm = Mx / 1000.0
My_kNm = My / 1000.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico para Mx (Momento que pide armadura en dirección X)
# Usamos contourf para rellenar de color entre las isolíneas
contour1 = ax1.contourf(X, Y, Mx_kNm, levels=15, cmap='inferno')
ax1.set_title('Momentos Flectores $M_x$ (kNm/m)', fontsize=13)
ax1.set_xlabel('Eje X (m)')
ax1.set_ylabel('Eje Y (m)')
ax1.set_aspect('equal') # Mantener proporción de la losa en 2D
fig.colorbar(contour1, ax=ax1, label='Momento (kNm/m)')

# Gráfico para My (Momento que pide armadura en dirección Y)
contour2 = ax2.contourf(X, Y, My_kNm, levels=15, cmap='inferno')
ax2.set_title('Momentos Flectores $M_y$ (kNm/m)', fontsize=13)
ax2.set_xlabel('Eje X (m)')
ax2.set_ylabel('Eje Y (m)')
ax2.set_aspect('equal')
fig.colorbar(contour2, ax=ax2, label='Momento (kNm/m)')

fig1 = plt.figure(figsize=(12, 8))
ax = fig1.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, -W_2D * 1000, cmap='plasma', edgecolor='k', linewidth=0.3, alpha=0.9)
ax.set_title('Deformada de Losa de Hormigón Simplemente Apoyada', fontsize=14, pad=20)
ax.set_box_aspect((Lx, Ly, 1.0))
fig1.colorbar(surf, shrink=0.5, aspect=10, label='Deformación (mm)')

plt.tight_layout()
plt.show()