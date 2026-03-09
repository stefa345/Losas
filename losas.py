import numpy as np
import scipy.sparse as sp

# 1. Parámetros geométricos y del material
Lx = 8.0        # Longitud en x (m)
Ly = 3.0        # Longitud en y (m)
h = 0.15        # Espesor de la losa (m)
E = 30e9        # Módulo de elasticidad del hormigón (Pa) -> 30 GPa
nu = 0.2        # Coeficiente de Poisson
q_load = 5000   # Carga transversal distribuida (N/m^2)

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