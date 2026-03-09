import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. Parámetros geométricos y del material
Lx = 12.0        # Longitud en x (m)
Ly = 3.0        # Longitud en y (m)
h = 0.15        # Espesor de la losa (m)
E = 30e9        # Módulo de elasticidad del hormigón (Pa) -> 30 GPa
nu = 0.2        # Coeficiente de Poisson
q_load = 8000   # Carga transversal distribuida (N/m^2)

# Rigidez a la flexión de la placa (D)
D = (E * h**3) / (12 * (1 - nu**2))

# 2. Discretización de la malla
delta = 0.25    # Tamaño del paso de la malla dx = dy = delta (m)
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

for j in range(ny_in):
    for i in range(nx_in):
        k = get_k(i, j)
        
        # 1. Coeficiente central base
        coef_central = 20.0
        
        # 2. Aplicamos la lógica de los nodos fantasmas para modificar el coeficiente central
        # Si estamos pegados a un borde (i=0, i=nx_in-1, j=0, j=ny_in-1), 
        # el nodo a distancia 2 cae afuera y se resta 1 al central.
        if i == 0: coef_central -= 1.0          # Borde izquierdo
        if i == nx_in - 1: coef_central -= 1.0  # Borde derecho
        if j == 0: coef_central -= 1.0          # Borde inferior
        if j == ny_in - 1: coef_central -= 1.0  # Borde superior
        
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
W_2D = np.zeros((Nx + 1, Ny + 1))

# Llenamos el interior de la matriz con la solución que acabamos de hallar
for j in range(ny_in):
    for i in range(nx_in):
        W_2D[i+1, j+1] = w_1D[get_k(i, j)]

# Extraemos la deformación máxima (en el centro de la losa)
flecha_max = np.max(W_2D)
print(f"Resolución completa. Flecha máxima en el centro: {flecha_max * 1000:.2f} mm")

# ==========================================
# 6. GRÁFICO 3D DE LA DEFORMADA
# ==========================================
# Creamos las coordenadas X e Y para el gráfico
x_coord = np.linspace(0, Lx, Nx + 1)
y_coord = np.linspace(0, Ly, Ny + 1)
X, Y = np.meshgrid(x_coord, y_coord, indexing='ij')

# Configuramos la figura
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Invertimos el eje Z para que la deformación se vea hacia abajo (como en la realidad)
surf = ax.plot_surface(X, Y, -W_2D*1000, cmap=cm.viridis, edgecolor='k', linewidth=0.5, alpha=0.9)

# Etiquetas y formato
ax.set_title('Deformada de Losa de Hormigón Simplemente Apoyada', fontsize=14)
ax.set_box_aspect((Lx, Ly, 0.5))

# Agregamos una barra de colores
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Desplazamiento (mm)')

plt.tight_layout()
plt.show()