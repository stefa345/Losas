import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. INICIALIZACIÓN Y PARÁMETROS
# ==========================================
ops.wipe()
# Modelo 3D con 6 Grados de Libertad por nodo
ops.model('basic', '-ndm', 3, '-ndf', 6)

# Parámetros sincronizados con losas_MDF.py
Lx = 3.0        # Longitud en x (m)
Ly = 5.0        # Longitud en y (m)
h = 0.10        # Espesor de la losa (m)
E = 30.67e9     # Módulo de elasticidad (Pa)
nu = 0.2        # Coeficiente de Poisson
q_load = -4000  # Carga distribuida (N/m^2) hacia abajo (-Z)

# Discretización (delta = 0.2m)
Nx = 15         # Elementos en X (3.0 / 0.2)
Ny = 25         # Elementos en Y (5.0 / 0.2)
dx = Lx / Nx
dy = Ly / Ny

# ==========================================
# 2. MATERIALES Y SECCIÓN
# ==========================================
matTag = 1
secTag = 1
# Material elástico isótropo
ops.nDMaterial('ElasticIsotropic', matTag, E, nu)
# Sección de placa elástica (E, nu, h, rho)
ops.section('ElasticMembranePlateSection', secTag, E, nu, h, 0.0)

# ==========================================
# 3. CREACIÓN DE MALLA (NODOS Y ELEMENTOS)
# ==========================================
node_tags = np.zeros((Ny + 1, Nx + 1), dtype=int)
tag = 1
for j in range(Ny + 1):
    y = j * dy
    for i in range(Nx + 1):
        x = i * dx
        ops.node(tag, x, y, 0.0)
        node_tags[j, i] = tag
        
        # CONDICIONES DE BORDE (Apoyo Simple)
        if i == 0 or i == Nx or j == 0 or j == Ny:
            # Fijamos Uz (grado 3)
            # Fijamos Ux (1) y Uy (2) solo para estabilidad global (evitar movimiento rígido)
            if i == 0 and j == 0:
                ops.fix(tag, 1, 1, 1, 0, 0, 0) # Esquina fija
            elif i == Nx and j == 0:
                ops.fix(tag, 0, 1, 1, 0, 0, 0) # Estabiliza rotación
            else:
                ops.fix(tag, 0, 0, 1, 0, 0, 0) # Resto: solo apoyo vertical
        tag += 1

ele_tag = 1
for j in range(Ny):
    for i in range(Nx):
        nI = int(node_tags[j, i])
        nJ = int(node_tags[j, i+1])
        nK = int(node_tags[j+1, i+1])
        nL = int(node_tags[j+1, i])
        # Elemento Shell de 4 nodos
        ops.element('ShellMITC4', ele_tag, nI, nJ, nK, nL, secTag)
        ele_tag += 1

# ==========================================
# 4. CARGAS (Tributarias)
# ==========================================
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
for j in range(Ny + 1):
    for i in range(Nx + 1):
        # Área tributaria: Nodo interno (dx*dy), Borde (dx*dy/2), Esquina (dx*dy/4)
        area = (dx*dy) / ( (1+(i==0 or i==Nx)) * (1+(j==0 or j==Ny)) )
        ops.load(int(node_tags[j,i]), 0.0, 0.0, q_load * area, 0.0, 0.0, 0.0)

# ==========================================
# 5. ANÁLISIS
# ==========================================
ops.constraints('Transformation')
ops.numberer('RCM')
ops.system('SparseGeneral')
ops.test('NormDispIncr', 1e-8, 10)
ops.algorithm('Newton')
ops.integrator('LoadControl', 1.0)
ops.analysis('Static')

print("Ejecutando análisis MEF en OpenSees...")
ops.analyze(1)

# ==========================================
# 6. EXTRACCIÓN DE RESULTADOS
# ==========================================
print("Extrayendo resultados...")

# A. Desplazamientos Nodales (Uz)
W_2D = np.zeros((Ny + 1, Nx + 1))
for j in range(Ny + 1):
    for i in range(Nx + 1):
        W_2D[j, i] = ops.nodeDisp(int(node_tags[j,i]), 3)

# B. Momentos Internos (Usando 'stresses' para resultantes de sección)
Mx_2D = np.zeros((Ny, Nx))
My_2D = np.zeros((Ny, Nx))

for e in range(1, Nx*Ny + 1):
    res = ops.eleResponse(e, 'stresses')
    # M11 (Flexión dir X - Índice 3, 11, 19, 27)
    # M22 (Flexión dir Y - Índice 4, 12, 20, 28)
    m11 = (res[3] + res[11] + res[19] + res[27]) / 4.0
    m22 = (res[4] + res[12] + res[20] + res[28]) / 4.0
    
    j_idx = (e-1) // Nx
    i_idx = (e-1) % Nx
    Mx_2D[j_idx, i_idx] = m11
    My_2D[j_idx, i_idx] = m22

# Estadísticas para comparación
w_max_mm = np.max(np.abs(W_2D)) * 1000.0
Mx_max_abs = np.max(np.abs(Mx_2D))
My_max_abs = np.max(np.abs(My_2D))

print("-" * 40)
print("COMPARATIVA FINAL CON MDF:")
print(f"-> Deflexión máx: {w_max_mm:.4f} mm")
print(f"-> Momento Mx máx: {Mx_max_abs:.2f} Nm/m")
print(f"-> Momento My máx: {My_max_abs:.2f} Nm/m")
print("-" * 40)

# ==========================================
# 7. VISUALIZACIÓN
# ==========================================
X_nod, Y_nod = np.meshgrid(np.linspace(0, Lx, Nx + 1), np.linspace(0, Ly, Ny + 1))
X_ele, Y_ele = np.meshgrid(np.linspace(dx/2, Lx - dx/2, Nx), np.linspace(dy/2, Ly - dy/2, Ny))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
c1 = ax1.contourf(X_ele, Y_ele, Mx_2D, levels=20, cmap='inferno')
fig.colorbar(c1, ax=ax1, label='Mx [Nm/m]')
ax1.set_title('Momentos Flectores Mx (MEF)')
ax1.set_aspect('equal')

c2 = ax2.contourf(X_ele, Y_ele, My_2D, levels=20, cmap='inferno')
fig.colorbar(c2, ax=ax2, label='My [Nm/m]')
ax2.set_title('Momentos Flectores My (MEF)')
ax2.set_aspect('equal')

fig3 = plt.figure(figsize=(10, 7))
ax3 = fig3.add_subplot(111, projection='3d')
surf = ax3.plot_surface(X_nod, Y_nod, W_2D*1000, cmap='plasma', edgecolor='k', linewidth=0.1)
ax3.set_title(f'Deformada MEF (Máx: {w_max_mm:.2f} mm)')
ax3.set_box_aspect((Lx, Ly, 1.0))
fig3.colorbar(surf, label='Uz [mm]')

plt.tight_layout()
plt.show()

ops.wipe()
