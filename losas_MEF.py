import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. INICIALIZACIÓN Y PARÁMETROS
# ==========================================
ops.wipe() # Limpia cualquier modelo previo en memoria
# Modelo 3D (ndm=3) con 6 grados de libertad por nodo (ndf=6)
ops.model('basic', '-ndm', 3, '-ndf', 6)

Lx = 3.0        # m
Ly = 5.0        # m
h = 0.10        # Espesor (m)
E = 30.67e9     # Módulo de elasticidad (Pa)
nu = 0.2        # Poisson
q_load = -4000  # Carga (N/m^2) -> Negativa porque va hacia abajo (eje Z)

# ==========================================
# 2. MATERIAL Y SECCIÓN
# ==========================================
matTag = 1
secTag = 1
# Material elástico isótropo
ops.nDMaterial('ElasticIsotropic', matTag, E, nu)
# Sección tipo placa/membrana elástica. El último 0.0 es la densidad (peso propio ignorado aquí)
ops.section('ElasticMembranePlateSection', secTag, E, nu, h, 0.0)

# ==========================================
# 3. GENERACIÓN DE MALLA (NODOS)
# ==========================================
Nx = 15
Ny = 25
dx = Lx / Nx
dy = Ly / Ny

# Matriz para guardar las etiquetas (tags) de los nodos y facilitar su llamado
node_tags = np.zeros((Nx+1, Ny+1), dtype=int)
tag = 1

for j in range(Ny + 1):
    y = j * dy
    for i in range(Nx + 1):
        x = i * dx
        # Crear nodo: ops.node(tag, x, y, z)
        ops.node(tag, x, y, 0.0)
        node_tags[i, j] = tag
        tag += 1

# ==========================================
# 4. CONDICIONES DE BORDE (APOYOS)
# ==========================================
# DOFs en OpenSees: 1=UX, 2=UY, 3=UZ, 4=RX, 5=RY, 6=RZ
# 1 significa bloqueado, 0 significa libre

for j in range(Ny + 1):
    for i in range(Nx + 1):
        nTag = int(node_tags[i, j])
        
        # Identificamos si el nodo está en algún borde perimetral
        if i == 0 or i == Nx or j == 0 or j == Ny:
            # Apoyo Simple: Bloqueamos traslaciones XYZ (para asimilar un muro que no cede)
            # Dejamos libres las rotaciones RX y RY para que la losa "gire" en el apoyo.
            # Bloqueamos RZ obligatoriamente.
            ops.fix(nTag, 1, 1, 1, 0, 0, 1)
        else:
            # Nodos internos de la losa: totalmente libres para hundirse y rotar, 
            # pero bloqueamos RZ (drilling DOF) por la formulación del Shell.
            ops.fix(nTag, 0, 0, 0, 0, 0, 1)

# ==========================================
# 5. GENERACIÓN DE ELEMENTOS (SHELL)
# ==========================================
eleTag = 1
for j in range(Ny):
    for i in range(Nx):
        # Tomamos los 4 nodos que forman un rectángulo (sentido antihorario)
        n1 = int(node_tags[i, j])
        n2 = int(node_tags[i+1, j])
        n3 = int(node_tags[i+1, j+1])
        n4 = int(node_tags[i, j+1])
        
        # Crear elemento ShellMITC4
        ops.element('ShellMITC4', eleTag, n1, n2, n3, n4, secTag)
        eleTag += 1

# ==========================================
# 6. APLICACIÓN DE CARGAS
# ==========================================
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)

# En MEF básico, la carga distribuida se transforma en cargas puntuales en los nodos
# usando el concepto de "Área Tributaria"
for j in range(Ny + 1):
    for i in range(Nx + 1):
        nTag = int(node_tags[i, j])
        area = dx * dy
        
        if (i == 0 or i == Nx) and (j == 0 or j == Ny):
            area = area / 4.0  # Nodos en las esquinas
        elif i == 0 or i == Nx or j == 0 or j == Ny:
            area = area / 2.0  # Nodos en los bordes
            
        Fz = q_load * area
        # Aplicamos fuerza: ops.load(nodo, Fx, Fy, Fz, Mx, My, Mz)
        ops.load(nTag, 0.0, 0.0, Fz, 0.0, 0.0, 0.0)

# ==========================================
# 7. RESOLUCIÓN DEL SISTEMA
# ==========================================
ops.system('BandSPD')               # Resolvedor de matrices simétricas
ops.numberer('RCM')                 # Optimizador de numeración para velocidad
ops.constraints('Transformation')   # Método para manejar los apoyos
ops.integrator('LoadControl', 1.0)  # Aplicar toda la carga en 1 paso
ops.algorithm('Linear')             # Comportamiento lineal elástico
ops.analysis('Static')              # Tipo de análisis: Estático

print("Resolviendo modelo en OpenSees...")
ops.analyze(1)
print("¡Análisis completado!")

# ==========================================
# 8. EXTRACCIÓN Y VISUALIZACIÓN DE RESULTADOS
# ==========================================
W_2D = np.zeros((Ny + 1, Nx + 1))

# Extraer el desplazamiento vertical (DOF 3) de cada nodo
for j in range(Ny + 1):
    for i in range(Nx + 1):
        nTag = int(node_tags[i, j])
        # Multiplicamos por 1000 para pasar de metros a milímetros
        # Usamos abs() porque el desplazamiento es negativo (hacia abajo)
        W_2D[j, i] = abs(ops.nodeDisp(nTag, 3)) * 1000

print(f"Desplazamiento vertical máximo en el centro: {np.max(W_2D):.4f} mm")

# Graficar
x_vals = np.linspace(0, Lx, Nx + 1)
y_vals = np.linspace(0, Ly, Ny + 1)
X, Y = np.meshgrid(x_vals, y_vals)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, W_2D, cmap='plasma', edgecolor='k', linewidth=0.2)
ax.set_title('Deformada (MEF - OpenSeesPy)', fontsize=14, pad=15)
ax.set_xlabel('Eje X (m)')
ax.set_ylabel('Eje Y (m)')
ax.set_box_aspect((Lx, Ly, 1.0))
ax.invert_zaxis()
fig.colorbar(surf, shrink=0.5, aspect=10, label='Desplazamiento [mm]')

plt.tight_layout()
plt.show()