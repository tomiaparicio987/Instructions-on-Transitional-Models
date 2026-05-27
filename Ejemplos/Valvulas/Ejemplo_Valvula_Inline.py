"""
Simulación: Cierre de Válvula en el medio del tramo
"""

import math
import Transient_Modules as tm
import importlib 
importlib.reload(tm)
import matplotlib.pyplot as plt

# 1. Cargar datos del archivo específico de la válvula
data = tm.Input('Input_Valvula_Inline.txt')

# 2. Definir el archivo de salida (coincidente con el OUTPUT del txt)
output_file = 'output_valvula_Inline.txt'

# 3. Definir componentes
pipe          = tm.Pipe(data.pipe_data)
output        = tm.Output(data.output_data)
# La librería maneja la válvula como un dispositivo en línea usando las mismas clases
device_inline = tm.DeviceInline(data.device_data, pipe)

# 4. Definir condiciones de contorno
ubc = tm.UpstreamBoundaryCondition(data.ubc_data)
dbc = tm.DownstreamBoundaryCondition(pipe, data.dbc_data)

# 5. Crear la Malla
device = device_inline.device
mesh   = tm.Mesh(pipe, ubc, dbc, output.nbr_of_els, device=device)
mesh[-1].head_constant = 0.0
# 6. Calcular Estado Estacionario inicial (flujo normal antes del cierre)
time = 0
mesh.steady_state()

mesh.output_printer(output_file, 'w')

# 7. Simular Estado Transitorio (Cierre de la válvula)
while time < output.simulation_time:
    # Aumenté la cantidad de decimales en el print porque tu time_step ahora es muy chico (0.000833)
    print(f'Time is {time:.6f} s') 
    time += mesh.delta_time
    mesh.extrapolate(time)
    mesh.output_printer(output_file, 'a')
    
print("Generando video de la animacion...")
tm.Viz(
    filename   = output_file,
    videofile  = 'Simulacion_Valvula_Inline.avi',
    x_max      = pipe.length,
    delta_time = mesh.delta_time,
    xlabel     = 'Length (m)',
    ylabel     = 'Head (m)',
    y2label    = 'Discharge (m3/s)',
    dpi_value  = 100,
)
print("Video guardado como 'Simulacion_Valvula_Inline.avi'")
plt.show()