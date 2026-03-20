import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARÁMETROS Y CONFIGURACIÓN DEL SISTEMA
# ==========================================
ops.wipe()
ops.model('basic', '-ndm', 3, '-ndf', 6)

# --- DEFINICIÓN DE TRAMOS ---
Lx_list = [4.0, 1.0]  # Luces en X (m)
Ly_list = [5.0, 5.0]       # Luces en Y (m)
Nx_list = [10, 10]     # Elementos por tramo en X
Ny_list = [12, 12]         # Elementos por tramo en Y

# --- PROPIEDADES ---
h = 0.15        # Espesor general (m)
E = 30.0e9      # Pa
nu = 0.2

# --- APOYOS EN EJES ---
# Condiciones: 'SIMPLE' (Uz), 'FIJO' (Empotrado), 'LIBRE'
# Se definen para cada línea de la parrilla (N_tramos + 1)
apoyos_X = ['FIJO', 'SIMPLE', 'LIBRE'] # Apoyos en x=0, x=4, x=9, x=13
apoyos_Y = ['SIMPLE', 'SIMPLE', 'SIMPLE']              # Apoyos en y=0, y=5, y=10 (ej: bordes libres)

# --- CARGAS POR TABLERO (N_tramos_Y x N_tramos_X) ---
q_loads = np.array([
    [-4000, -4000],  # Tableros fila inferior
    [-4000, -4000]   # Tableros fila superior (ej: carga alterna)
])

# Patrones de BC
patrones_bc = {
    'SIMPLE': [0, 0, 1, 0, 0, 0],
    'FIJO':   [1, 1, 1, 1, 1, 1],
    'LIBRE':  [0, 0, 0, 0, 0, 0]
}

# ==========================================
# 2. GENERACIÓN DE COORDENADAS DE RED
# ==========================================
def generate_coords(span_list, div_list):
    coords = [0.0]
    for L, N in zip(span_list, div_list):
        last = coords[-1]
        new_coords = np.linspace(last, last + L, N + 1)[1:]
        coords.extend(new_coords)
    return np.array(coords)

x_coords = generate_coords(Lx_list, Nx_list)
y_coords = generate_coords(Ly_list, Ny_list)

total_Nx = sum(Nx_list)
total_Ny = sum(Ny_list)

# Indices de los ejes (grid lines)
grid_indices_X = [0] + list(np.cumsum(Nx_list))
grid_indices_Y = [0] + list(np.cumsum(Ny_list))

# ==========================================
# 3. MATERIALES Y SECCIÓN
# ==========================================
matTag, secTag = 1, 1
ops.nDMaterial('ElasticIsotropic', matTag, E, nu)
ops.section('ElasticMembranePlateSection', secTag, E, nu, h, 0.0)

# ==========================================
# 4. CREACIÓN DE NODOS Y BCs
# ==========================================
node_tags = np.zeros((total_Ny + 1, total_Nx + 1), dtype=int)
tag = 1
for j in range(total_Ny + 1):
    y = y_coords[j]
    for i in range(total_Nx + 1):
        x = x_coords[i]
        ops.node(tag, x, y, 0.0)
        node_tags[j, i] = tag
        
        # Lógica de BCs
        res = [0] * 6
        
        # Verificar si el nodo está en un eje vertical (X fijo)
        if i in grid_indices_X:
            idx = grid_indices_X.index(i)
            cond = apoyos_X[idx]
            res = [a | b for a, b in zip(res, patrones_bc[cond])]
            
        # Verificar si el nodo está en un eje horizontal (Y fijo)
        if j in grid_indices_Y:
            idx = grid_indices_Y.index(j)
            cond = apoyos_Y[idx]
            res = [a | b for a, b in zip(res, patrones_bc[cond])]
            
        # Estabilidad Global
        if i == 0 and j == 0:
            res[0] = 1; res[1] = 1
        elif i == total_Nx and j == 0:
            res[1] = 1
            
        if any(res):
            ops.fix(tag, *res)
        tag += 1

# ==========================================
# 5. CREACIÓN DE ELEMENTOS Y CARGAS
# ==========================================
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)

ele_tag = 1
# Para el cálculo de cargas tributarias, necesitamos las dimensiones de cada celda
dx_vals = np.diff(x_coords)
dy_vals = np.diff(y_coords)

# Mapeo de elemento a tablero para carga
for j in range(total_Ny):
    # Encontrar índice de tramo en Y
    y_span_idx = next(idx for idx, val in enumerate(grid_indices_Y[1:]) if j < val)
    for i in range(total_Nx):
        # Encontrar índice de tramo en X
        x_span_idx = next(idx for idx, val in enumerate(grid_indices_X[1:]) if i < val)
        
        nI, nJ, nK, nL = int(node_tags[j,i]), int(node_tags[j,i+1]), int(node_tags[j+1,i+1]), int(node_tags[j+1,i])
        ops.element('ShellMITC4', ele_tag, nI, nJ, nK, nL, secTag)
        
        # Carga del tablero correspondiente
        q = q_loads[y_span_idx, x_span_idx]
        area = dx_vals[i] * dy_vals[j]
        # Distribución simple a los 4 nodos
        for node in [nI, nJ, nK, nL]:
            ops.load(node, 0.0, 0.0, q * area / 4.0, 0.0, 0.0, 0.0)
            
        ele_tag += 1

# ==========================================
# 6. ANÁLISIS
# ==========================================
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('SparseGeneral')
ops.test('NormDispIncr', 1e-8, 10)
ops.algorithm('Newton')
ops.integrator('LoadControl', 1.0)
ops.analysis('Static')

print("Ejecutando análisis de losas continuas...")
ops.analyze(1)

# ==========================================
# 7. EXTRACCIÓN Y VISUALIZACIÓN
# ==========================================
print("Extrayendo resultados...")

# A. Desplazamientos (Nodales)
W_2D = np.zeros((total_Ny + 1, total_Nx + 1))
for j in range(total_Ny + 1):
    for i in range(total_Nx + 1):
        W_2D[j, i] = ops.nodeDisp(int(node_tags[j,i]), 3)

print('-'*40)
print(f"Deflexión máx: {np.max(W_2D)*1000:.4f} mm")
print(f"Deflexión mín: {np.min(W_2D)*1000:.4f} mm")
print('-'*40)

# B. Momentos Internos (Por elemento)
Mx_2D = np.zeros((total_Ny, total_Nx))
My_2D = np.zeros((total_Ny, total_Nx))

for e in range(1, total_Nx * total_Ny + 1):
    try:
        res = ops.eleResponse(e, 'stresses')
        # M11 (Flexión dir X) y M22 (Flexión dir Y)
        # Promedio de los 4 puntos de integración/nodos
        m11 = -(res[3] + res[11] + res[19] + res[27]) / 4.0
        m22 = -(res[4] + res[12] + res[20] + res[28]) / 4.0
        
        j_idx = (e-1) // total_Nx
        i_idx = (e-1) % total_Nx
        Mx_2D[j_idx, i_idx] = m11
        My_2D[j_idx, i_idx] = m22
    except:
        continue

# Coordenadas para mallas
X_mesh, Y_mesh = np.meshgrid(x_coords, y_coords)
# Centros de elementos para momentos
x_centers = (x_coords[:-1] + x_coords[1:]) / 2
y_centers = (y_coords[:-1] + y_coords[1:]) / 2
X_ele, Y_ele = np.meshgrid(x_centers, y_centers)

# --- GRÁFICOS ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Mapa de Mx
c1 = axes[0].contourf(X_ele, Y_ele, Mx_2D, levels=40, cmap='inferno')
fig.colorbar(c1, ax=axes[0], label='Mx [Nm/m]')
axes[0].set_title('Momentos Mx (Dirección X)')
axes[0].set_xlabel('X [m]')
axes[0].set_ylabel('Y [m]')

# Mapa de My
c2 = axes[1].contourf(X_ele, Y_ele, My_2D, levels=40, cmap='inferno')
fig.colorbar(c2, ax=axes[1], label='My [Nm/m]')
axes[1].set_title('Momentos My (Dirección Y)')
axes[1].set_xlabel('X [m]')
axes[1].set_ylabel('Y [m]')

# Añadir líneas de apoyos a ambos gráficos
for ax in axes:
    for xc in [sum(Lx_list[:i]) for i in range(1, len(Lx_list))]:
        ax.axvline(xc, color='k', linestyle='--', alpha=0.5)
    for yc in [sum(Ly_list[:i]) for i in range(1, len(Ly_list))]:
        ax.axhline(yc, color='k', linestyle='--', alpha=0.5)
    ax.set_aspect('equal')

# Gráfico de Deflexión aparte (3D)
fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111, projection='3d')
surf = ax3.plot_surface(X_mesh, Y_mesh, W_2D*1000, cmap='plasma', edgecolor='k', linewidth=0.1)
ax3.set_title(f'Deformada MEF (Máx: {np.max(W_2D)*1000:.4f} mm) (Min: {np.min(W_2D)*1000:.4f} mm)')
ax3.set_box_aspect((np.max(x_coords), np.max(y_coords), 1.0))
fig3.colorbar(surf, label='Uz [mm]')

plt.tight_layout()
plt.show()

ops.wipe()
