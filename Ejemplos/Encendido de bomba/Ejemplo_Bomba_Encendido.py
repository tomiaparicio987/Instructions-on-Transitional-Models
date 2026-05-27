# -*- coding: utf-8 -*-
"""
Simulación: Encendido de una bomba.
"""

import math
import Transient_Modules as tm
import importlib 
importlib.reload(tm)
import matplotlib.pyplot as plt

# 1. Cargar datos del archivo
data = tm.Input('Input_BombaEncendido.txt')



output_file = 'output_Bomba_Encendido.txt'



# 2. Definir componentes

pipe          = tm.Pipe(data.pipe_data)
output        = tm.Output(data.output_data)
device_inline = tm.DeviceInline(data.device_data, pipe)

# 3. Definir condiciones de contorno

ubc = tm.UpstreamBoundaryCondition(data.ubc_data)
dbc = tm.DownstreamBoundaryCondition(pipe, data.dbc_data)

# 4. Crear la Malla

device = device_inline.device
mesh   = tm.Mesh(pipe, ubc, dbc, output.nbr_of_els, device=device)

# 5. Calcular Estado Estacionario

mesh.steady_state()

#Steady state
time = 0

mesh.output_printer(output_file, 'w')

#Unsteady state
while time < output.simulation_time:
    print(f'Time is {time:.2f} s')
    time += mesh.delta_time
    mesh.extrapolate(time)
    mesh.output_printer(output_file, 'a')
    
print("Generando video de la animacion...")
tm.Viz(
    filename   = output_file,
    videofile  = 'Simulacion_Bomba_Encendido.avi',
    x_max      = pipe.length,
    delta_time = mesh.delta_time,
    xlabel     = 'Lenght (m)',
    ylabel     = 'Head (m)',
    y2label    = 'Discharge (m3/s)',
    dpi_value  = 100,
)
print("Video guardado como 'Simulacion_Bomba_Encendido.avi'")
plt.show()
