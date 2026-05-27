# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:22:22 2026

@author: Usuario
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from numpy import argmin  
from scipy.optimize import root
from scipy.optimize import fsolve                                            

class Const:
    G = 9.81
    maxIter = 100
    tolerance = 1.00E-06
    Hatm = 10.33
    N = 1.40
    
class Input:
    def __init__(self, inputfile_name):
        self.inputfile_name = inputfile_name
    
        with open(self.inputfile_name, 'r') as f:
            data_blocks = f.read().replace('\r\n', '\n').replace('\r', '\n').replace('\n \n', '\n\n').split('\n\n')
        print(f"Bloques encontrados: {len(data_blocks)}")
        for i, block in enumerate(data_blocks):
            print(f"  Bloque {i}: {repr(block.split(chr(10))[0])}")
        
        self.complementary_data = {}
        main_block_counter = 0
        for block in data_blocks:
            header = block.split('\n')[0]
            if header == 'PIPE':
                self.pipe_data = self.read_data_block_as_dict(block.split('\n')[1:])
                main_block_counter += 1	
            elif header == 'OUTPUT':
                self.output_data = self.read_data_block_as_dict(block.split('\n')[1:])
                main_block_counter += 1
            elif header == 'DEVICES':
                device_inline = block.split('\n')[1]
                main_block_counter += 1
            elif header == 'UPSTREAM_BOUNDARY_CONDITION':
                ubc_device = block.split('\n')[1]
                main_block_counter += 1
            elif header == 'DOWNSTREAM_BOUNDARY_CONDITION':
                dbc_device = block.split('\n')[1]
                main_block_counter += 1
            else:
                self.complementary_data[header] = block.split('\n')[1:]
        if main_block_counter == 5:
            print('All main blocks were found')
        else:
            raise Exception('One main block is missing')

        #Chequear por los elementos en línea
        self.device_data = {}
        if device_inline != 'none':
#            self.device_data = {}
            self.device_data['position'] = device_inline.split()[1]     #Aquí debería contemplar la posibilidad de que no haya posición en el input file
            if device_inline.split()[0] == 'valve':
	            if 'VALVE_INLINE' not in self.complementary_data.keys():
		            raise Exception('The VALVE_INLINE block is missing')
	            else:
	                self.device_data['valve'] = self.read_data_block_as_dict(self.complementary_data['VALVE_INLINE'])
            elif device_inline.split()[0] == 'pump':
	            if 'PUMP_INLINE' not in self.complementary_data.keys():
		            raise Exception('The PUMP_INLINE block is missing')
	            else:
	                self.device_data['pump'] = self.read_data_block_as_dict(self.complementary_data['PUMP_INLINE'])	
            elif device_inline.split()[0] == 'surge_tank':
	            if 'SURGE_TANK' not in self.complementary_data.keys():
		            raise Exception('The SURGE_TANK block is missing')
	            else:
	                self.device_data['surge_tank'] = self.read_data_block_as_dict(self.complementary_data['SURGE_TANK'])	
            elif device_inline.split()[0] == 'turbopump':
	            if 'TURBOPUMP' not in self.complementary_data.keys():
		            raise Exception('The TURBOPUMP block is missing')
	            else:
	                self.device_data['turbopump'] = self.read_data_block_as_dict(self.complementary_data['TURBOPUMP'])	
            elif device_inline.split()[0] == 'air_chamber':
	            if 'AIR_CHAMBER' not in self.complementary_data.keys():
		            raise Exception('The AIR_CHAMBER block is missing')
	            else:
	                self.device_data['air_chamber'] = self.read_data_block_as_dict(self.complementary_data['AIR_CHAMBER'])	
            else:
                raise Exception('There was a problem while reading the DEVICES block')
        
        #Chequear por elementos en la frontera de aguas arriba
        self.ubc_data = {}
        if ubc_device == 'reservoir':
            if 'RESERVOIR' not in self.complementary_data.keys():
                raise Exception('The RESERVOIR block is missing')
            else:
                self.ubc_data['reservoir'] = self.read_data_block_as_dict(self.complementary_data['RESERVOIR'])
        elif ubc_device == 'pump':
            if 'PUMP_UPSTREAM' not in self.complementary_data.keys():
                raise Exception('The PUMP_UPSTREAM block is missing')
            else:
                self.ubc_data['pump'] = self.read_data_block_as_dict(self.complementary_data['PUMP_UPSTREAM'])
        else:
            raise Exception('There was a problem while reading the UPSTREAM_BOUNDARY_CONDITION block')

        #Chequear por los elementos en la frontera de aguas abajo
        self.dbc_data = {}
        if dbc_device == 'valve':
            if 'VALVE_END' not in self.complementary_data.keys():
                raise Exception('The VALVE_END block is missing')
            else:
                self.dbc_data['valve'] = self.read_data_block_as_dict(self.complementary_data['VALVE_END'])
        elif dbc_device == 'turbine':
            if 'TURBINE' not in self.complementary_data.keys():
                raise Exception('The TURBINE block is missing')
            else:
                self.dbc_data['turbine'] = self.read_data_block_as_dict(self.complementary_data['TURBINE'])	
        elif dbc_device == 'tailwater':
	        if 'TAILWATER' not in self.complementary_data.keys():
		        raise Exception('The TAILWATER block is missing')
	        else:
	            self.dbc_data['tailwater'] = self.read_data_block_as_dict(self.complementary_data['TAILWATER'])	
        else:
            raise Exception('There was a problem while reading the DOWNSTREAM_BOUNDARY_CONDITION block')

    def read_data_block_as_dict(self, content):
     
        block_data = {}    
        for line in content:
            data = line.split()
            if not data:          
                continue
            if len(data) >= 2:
                block_data[data[0]] = data[1]
        return block_data
    
class DownstreamBoundaryCondition:
    def __init__(self, pipe, read_dbc_data):
        name = list(read_dbc_data.keys())[0]
        if name == 'valve':
            required_data = ['valve_closing_start', 'valve_closing_end',
                         'valve_discharge_coefficient', 'valve_closing_exponent']
            for item in required_data:
                if item not in list(list(read_dbc_data.values())[0].keys()):
                    raise Exception('There was a problem while reading the VALVE_END block')
            valve_end_data = list(read_dbc_data.values())[0]
            self.device = Valve(pipe.length, None, None, pipe, valve_end_data)
        
        elif name == 'turbine':
             required_data = ['nominal_head', 'nominal_discharge', 'nominal_speed',
                              'nominal_torque', 'initial_power', 'power_variation', 'start_time',
                              'end_time', 'weight', 'gyration_radius', 'dashpot_time_constant',
                              'speed_servomotor_ratio', 'sigma', 'delta', 'dimensionless_data']
             for item in required_data:
                 if item not in list(list(read_dbc_data.values())[0].keys()):
                     raise Exception('There was a problem while reading the TURBINE block')
             turbine_data = list(read_dbc_data.values())[0]
             self.device = Turbine(pipe.length, None, None, turbine_data)
        # elif name == 'turbine':
        #     required_data = ['nominal_head', 'nominal_discharge', 'nominal_speed',
        #                      'nominal_torque', 'initial_power', 'power_variation', 'start_time', 'end_time',
        #                      'weight', 'gyration_radius', 'dashpot_time_constant',
        #                      'speed_servomotor_ratio', 'sigma', 'delta', 'dimensionless_data']
        #     for item in required_data:
        #         if item not in list(list(read_dbc_data.values())[0].keys()):
        #             raise Exception('There was a problem while reading the TURBINE block')
        #     turbine_data = list(read_dbc_data.values())[0]
        #     self.device = Turbine(None, None, None, turbine_data)
        elif name == 'tailwater':
            required_data = 'water_level'
            if required_data not in list(list(read_dbc_data.values())[0].keys()):
	            raise Exception('There was a problem while reading the TAILWATER block')
            tailwater_data = list(read_dbc_data.values())[0]
            self.device = Tailwater(pipe.length, tailwater_data)
#         else:
#            raise Exception('There was a problem while reading the DOWNSTREAM_BOUNDARY_CONDITION block')

class UpstreamBoundaryCondition:
    def __init__(self, read_ubc_data):
        name = list(read_ubc_data.keys())[0]
        if name == 'reservoir':
            required_data = 'water_level'
            if required_data not in list(list(read_ubc_data.values())[0].keys()):
	            raise Exception('There was a problem while reading the RESERVOIR block')
            reservoir_data = list(read_ubc_data.values())[0]
            self.device = Reservoir(0., reservoir_data)
        #elif name == 'pump':
        #    self.device = Pump()
        else:
            raise Exception('There was a problem while reading the UPSTREAM_BOUNDARY_CONDITION block')
      
class Node:
    def __init__(self, x):
        self.x = x
        self.old_head = None
        self.old_discharge = None
        self.new_head = None
        self.new_discharge = None
        
    def __repr__(self):
        return f'{self.x:10.4f} {self.old_head[0]:10.4f} {self.old_discharge[0]:10.4f}'

    def setInit(self, head,discharge):
        self.setOld([head], [discharge])

    def setOld(self, head, discharge):
        self.old_head = head
        self.old_discharge = discharge
    
    def setNew(self, head, discharge):
        self.new_head = [head]
        self.new_discharge = [discharge]

    def solve(self, b, cp, cm, time):
        self.setNew(0.5*(cp + cm), 0.5*(cp - cm)/b)

class Reservoir(Node):
    def __init__(self, x, read_reservoir_data):
        super().__init__(x or 0.)
        self.water_level = float(read_reservoir_data['water_level'])
        self.setInit(self.water_level, 0.0)
        
    def solveAsUbc(self, b, cm, time):
        self.setNew(self.water_level, (self.water_level - cm)/b)
        
    def solveAsDbc (self, b, cp, time):                                    
        self.setNew(-self.water_level, (self.water_level + cp)/b)          

    def steady_state_term(self, discharge):
        return self.water_level

    def diff_steady_state_term(self, discharge):
        return 0.

class Valve(Node):
    def __init__(self, x, head, discharge, pipe, data):
        super().__init__(x)
        self.diameter = pipe.diameter
        self.coefficient = float(data['valve_discharge_coefficient'])
        self.closing_start = float(data['valve_closing_start'])
        self.closing_end = float(data['valve_closing_end'])
        self.exponent = float(data['valve_closing_exponent'])
        
        self.area = (math.pi/4.)*self.diameter**2
        self.cvp = Const.G*(self.coefficient*self.area)**2
    
    def __repr__(self):
        return (f'{self.x:10.4f} {self.old_head[0]:10.4f} {self.old_discharge[0]:10.4f}\n'
                f'{self.x:10.4f} {self.old_head[1]:10.4f} {self.old_discharge[0]:10.4f}')
    
    def setInit(self, *args):
        if len(args) == 2:
            head, discharge = args
            head_downstream = head + self.steady_state_term(discharge)
            self.setOld([head, head_downstream], [discharge])
            
        elif len(args) == 3:
            head_upstream, head_downstream, discharge = args
            self.setOld([head_upstream, head_downstream], [discharge])                           

    def setNew(self, head_upstream, head_downstream, discharge):
        self.new_head = [head_upstream, head_downstream]
        self.new_discharge = [discharge]

    def steady_state_term(self, discharge):
        return -(1./(2.*Const.G*(self.coefficient*self.area)**2))*discharge*abs(discharge)

    def diff_steady_state_term(self, discharge):
        return -(1./(Const.G*(self.coefficient*self.area)**2))*abs(discharge)

    def cv(self, time):
        if time <= self.closing_start:
            tau = 1.
            cv = self.cvp
        elif time > self.closing_start and time <= self.closing_end:
            tau = (1. - (time - self.closing_start)/(self.closing_end - self.closing_start))**self.exponent
            cv = (tau**2)*self.cvp
        else:
            tau = 0.
            cv = 0.
        return cv

    def solve(self, b, cp, cm, time):                                                                           #new
        if cp - cm > 0.:                                                                                        #new
		#Positive flow                                                                                          #new
            discharge = -2.*self.cv(time)*b + math.sqrt((2.*self.cv(time)*b)**2 + 2.*self.cv(time)*(cp - cm))        #new
        else:                                                                                                   #new
		#Negative flow                                                                                          #new
            discharge = 2.*self.cv(time)*b - math.sqrt((2.*self.cv(time)*b)**2 + 2.*self.cv(time)*(cp - cm))         #new
        self.setNew(cp - b*discharge, cm + b*discharge, discharge)                                                             #new

    def solveAsDbc(self, b, cp, time):
        if cp < 0.:                                                                                             #new
            discharge = 0.                                                                                           #new
        else:                                                                                                   #new
            discharge = -self.cv(time)*b + math.sqrt((self.cv(time)*b)**2 + 2.*self.cv(time)*cp)                     #new
        self.setNew(cp - b*discharge, 0., discharge)


class Pipe:
    def __init__(self, read_pipe_data):
        required_data = ['length', 'diameter', 'friction_coefficient', 'wave_speed', 'thickness', 'no_flow']
        for item in required_data:
            if item not in read_pipe_data.keys():
                raise Exception('There was a problem while reading the PIPE block')
        self.length = float(read_pipe_data['length'])
        self.diameter = float(read_pipe_data['diameter'])
        self.friction_coefficient = float(read_pipe_data['friction_coefficient'])
        self.wave_speed = float(read_pipe_data['wave_speed'])
        self.thickness = float(read_pipe_data['thickness'])
        no_flow_string = read_pipe_data['no_flow']
        if no_flow_string == 'True':
            self.no_flow = True
        elif no_flow_string == 'False':
            self.no_flow = False
        else:
            raise Exception('no-flow can only be either True or False in PIPE block')
        self.area = (math.pi/4.)*self.diameter**2    

    def steady_state_term(self, discharge_0):
        return -(self.friction_coefficient*self.length/(2.*Const.G*self.diameter*self.area**2))*discharge_0**2

    def diff_steady_state_term(self, discharge_0):
        return -(self.friction_coefficient*self.length/(Const.G*self.diameter*self.area**2))*discharge_0

class Mesh(list):
    def __init__(self, pipe, ubc, dbc, nbr_of_els, device=None):              #new
        self.nbr_of_els = nbr_of_els
        self.pipe = pipe
        self.delta_x = self.pipe.length/self.nbr_of_els
        self.r = self.pipe.friction_coefficient*self.delta_x/(2*Const.G*pipe.diameter*pipe.area**2)
        self.b = self.pipe.wave_speed/(Const.G*pipe.area)
        self.delta_time = self.delta_x/self.pipe.wave_speed
        self.device_node = None
        
                                                               #new
        
        self.append(ubc.device)
        for i in range(self.nbr_of_els - 1):
            self.append(Node(x = (i + 1)*self.delta_x))
        self.append(dbc.device)
        
        if device == None:
            self.device_node = None
        else:
            distance = [abs(device.x - i*self.delta_x) for i in range(self.nbr_of_els + 1)]     
            self.device_node = argmin(distance)                                                 
            self[self.device_node] = device 

    def steady_state(self):
        device = self[self.device_node] if self.device_node is not None else None
        device_state = device.state if (device is not None and hasattr(device, 'state')) else 0

        if device_state == 1:
            # Startup: bomba arranca apagada, caudal inicial = 0
            discharge_0 = 0.
        else:
            # Shutdown o SurgeTank: resolver Newton-Raphson para caudal inicial
            if hasattr(self[-1], 'nominal_discharge'):
               discharge_0 = self[-1].nominal_discharge
            else:
               discharge_0 = self.pipe.area
            
            
            for n in range(Const.maxIter):
                fx = self.pipe.steady_state_term(discharge_0) \
                + self[0].steady_state_term(discharge_0) \
                + self[-1].steady_state_term(discharge_0)
                df_dx = self.pipe.diff_steady_state_term(discharge_0) \
                + self[0].diff_steady_state_term(discharge_0) \
                + self[-1].diff_steady_state_term(discharge_0)

                if self.device_node is not None:
                    fx += self[self.device_node].steady_state_term(discharge_0)
                    df_dx += self[self.device_node].diff_steady_state_term(discharge_0)

                if abs(fx) < Const.tolerance:
                    break
                if df_dx == 0:
                    raise Exception('Zero derivative was reached when attempting to solve for steady-state conditions')
                discharge_0 = discharge_0 - fx/df_dx
                if n == Const.maxIter - 1:
                    raise Exception('Convergence failed while attempting to solve for steady-state conditions')
        print(discharge_0)
        print(f'NR iter {n}: Q={discharge_0:.4f}, fx={fx:.6f}, df={df_dx:.6f}')
        delta_head = self.r * discharge_0**2

        # ── Nodo aguas arriba (UBC) ──────────────────────────────────────────
        head = self[0].steady_state_term(discharge_0)
        self[0].setInit(head, discharge_0)

        if self.device_node is None:
            # 1. Inicializar todos los nodos internos (menos el último)
            for i in range(1, self.nbr_of_els):
                head = self[i - 1].old_head[-1] - delta_head
                self[i].setInit(head, discharge_0)
            
            # 2. Inicializar el último nodo (DBC) considerando si es una Turbina
            head = self[self.nbr_of_els - 1].old_head[-1] - delta_head
            if hasattr(self[-1], 'nominal_head'):
                # Turbina: setInit propio inicializa [head_upstream, head_upstream - nominal_head]
                self[-1].setInit(head, discharge_0)
            else:
                # Nodo de contorno estándar (Reservoir, etc.)
                self[-1].setInit(head, discharge_0)

        elif isinstance(device, Pump):
      
            for i in range(1, self.device_node):
                head = self[i - 1].old_head[-1] - delta_head
                self[i].setInit(head, discharge_0)

            # 2. Inicializar DBC y propagar hacia atrás hasta el nodo posterior al device
            head = -self[-1].steady_state_term(discharge_0)
            self[-1].setInit(head, discharge_0)

            for i in range(self.nbr_of_els - 1, self.device_node, -1):
                head = self[i + 1].old_head[0] + delta_head
                self[i].setInit(head, discharge_0)

            # 3. Cabezas de la bomba calculadas desde cada extremo conocido
            head_upstream   = self[0].water_level - self.device_node * delta_head
            head_downstream = -self[-1].steady_state_term(discharge_0) \
                              + (self.nbr_of_els - self.device_node) * delta_head
            self[self.device_node].setInit(head_upstream, head_downstream, discharge_0)

        elif isinstance(device, SurgeTank):
         
            for i in range(1, self.nbr_of_els + 1):
                head = self[i - 1].old_head[-1] - delta_head
                self[i].setInit(head, discharge_0)

        elif isinstance(device, Valve):
           
            dbc_node = self[-1]
            if hasattr(dbc_node, 'water_level'):
               
                head = dbc_node.water_level
                self[-1].setInit(head, discharge_0)
                for i in range(self.nbr_of_els - 1, self.device_node, -1):
                    head = self[i + 1].old_head[0] + delta_head
                    self[i].setInit(head, discharge_0)
                # Propagar hacia adelante desde UBC hasta device_node-1
                for i in range(1, self.device_node):
                    head = self[i - 1].old_head[-1] - delta_head
                    self[i].setInit(head, discharge_0)
              
                head_upstream   = self[0].water_level - self.device_node * delta_head
                head_downstream = dbc_node.water_level + (self.nbr_of_els - self.device_node) * delta_head
                self[self.device_node].setInit(head_upstream, head_downstream, discharge_0)
            else:
                
                for i in range(1, self.device_node):
                    head = self[i - 1].old_head[-1] - delta_head
                    self[i].setInit(head, discharge_0)
                # Nodo de la válvula inline: setInit con dos heads
                head_upstream = self[self.device_node - 1].old_head[-1] - delta_head
                self[self.device_node].setInit(head_upstream, discharge_0)
                # Continuar desde aguas abajo de la válvula inline hasta DBC
                for i in range(self.device_node + 1, self.nbr_of_els + 1):
                    head = self[i - 1].old_head[-1] - delta_head
                    self[i].setInit(head, discharge_0)

        else:
            # ── Otro device inline genérico (AirChamber, etc.) ───────────────
            head = -self[-1].steady_state_term(discharge_0)
            self[-1].setInit(head, discharge_0)
            for i in range(self.nbr_of_els - 1, self.device_node, -1):
                head = self[i + 1].old_head[0] + delta_head
                self[i].setInit(head, discharge_0)
            for i in range(1, self.device_node):
                head = self[i - 1].old_head[-1] - delta_head
                self[i].setInit(head, discharge_0)
            head_upstream   = self[self.device_node - 1].old_head[-1] - delta_head
            head_downstream = head_upstream + device.steady_state_term(discharge_0)
            self[self.device_node].setInit(head_upstream, head_downstream, discharge_0)
             
    def extrapolate(self, time):
        print(f'Time is {time:.2f} s')
        # Upstream boundary condition
        cm = self[1].old_head[0] - self[1].old_discharge[0]*(self.b - self.r*abs(self[1].old_discharge[0]))
        self[0].solveAsUbc(self.b, cm, time)
		
		#Inner nodes
        for i in range (1, len(self) - 1):
            cp = self[i - 1].old_head[-1] + self[i - 1].old_discharge[-1]*(self.b - self.r*abs(self[i - 1].old_discharge[-1]))
            cm = self[i + 1].old_head[0] - self[i + 1].old_discharge[0]*(self.b - self.r*abs(self[i + 1].old_discharge[0]))
            self[i].solve(self.b, cp, cm, time)
		
		#Downstream boundary conditions
        cp = self[-2].old_head[-1] + self[-2].old_discharge[-1]*(self.b - self.r*abs(self[-2].old_discharge[-1]))		
        if hasattr(self[-1], 'nominal_head'):
           self[-1].solveAsDbc(self, cp, time)
        else:
           self[-1].solveAsDbc(self.b, cp, time)
        
        self.update()

    def update(self):
    	for node in self:
    		node.setOld(node.new_head, node.new_discharge)

    def output_printer(self, outputfile, mode):
        with open(outputfile, mode) as output:
            for node in self:
                output.write(f'{node} \n')
            output.write('\n')

class Pump(Node):

    def __init__(self, x, head, discharge, pipe, read_pump_data):
        super().__init__(x = 0.)
        self.mode = int(read_pump_data['startup-shutdown'])
        if self.mode != 1 and self.mode != 2:
            raise Exception('startup-shutdown should equal either 1 or 2 in PUMP block')
        if self.mode == 2 and pipe.no_flow:
            raise Exception('startup-shutdown = 2 is not compatible with no-flow conditions')
        self.start_time = float(read_pump_data['start_time'])
        self.end_time = float(read_pump_data['end_time'])
        if self.start_time > self.end_time:
            raise Exception('end_time should be greater than start_time in PUMP block')
        self.hs = float(read_pump_data['shutoff_head'])
        self.threshold = float(read_pump_data['threshold_head'])
        if self.threshold > self.hs:
            raise Exception('shutoff_head should be greater than threshold_head in PUMP block')
        self.a1 = float(read_pump_data['linear_term_coefficient'])
        self.a2 = float(read_pump_data['quadratic_term_coefficient'])
        self.downstream_head = None
        self.downstream_discharge = None
        # state=0: bomba encendida (mode=1 shutdown, arranca ON)
        # state=1: bomba apagada  (mode=2 startup,  arranca OFF)
        self.state = 0 if self.mode == 1 else 1


    def __repr__(self):
       eps = 0.01
       return f'{self.x - eps:10.4f} {self.old_head[0]:10.4f} {self.old_discharge[0]:10.4f}\n{self.x + eps:10.4f} {self.old_head[1]:10.4f} {self.old_discharge[0]:10.4f}'

    def setInit(self, head_up, head_down, discharge):
        self.setOld([head_up, head_down], [discharge])

    def setNew(self, head_upstream, head_downstream, discharge):
        self.new_head = [head_upstream, head_downstream]
        self.new_discharge = [discharge]

    def steady_state_term(self, discharge_0):
        if self.state == 0:   # encendida: aporta su curva al balance
            return self.hs + self.a1*discharge_0 + self.a2*discharge_0**2
        else:                 # apagada: no aporta nada
            return 0.

    def diff_steady_state_term(self, discharge_0):
        if self.state == 0:
            return self.a1 + 2.*self.a2*discharge_0
        else:
            return 0.

    def alpha(self, time):
        if self.state == 0:   # shutdown: alpha 1 → 0
            if time <= self.start_time:
                alpha = 1.
            elif time < self.end_time:
                alpha = 1. - (time - self.start_time)/(self.end_time - self.start_time)
            else:
                alpha = 0.
        else:                 # startup: alpha 0 → 1
            if time <= self.start_time:
                alpha = 0.
            elif time < self.end_time:
                alpha = (time - self.start_time)/(self.end_time - self.start_time)
            else:
                alpha = 1.
        return alpha

    def solve(self, b, cp, cm, time):
        const_a = self.a2
        const_b = self.a1*self.alpha(time) - 2.*b
        const_c = self.hs*self.alpha(time)**2 + cp - cm
        discharge = (-const_b - math.sqrt(const_b**2 - 4.*const_a*const_c))/(2.*const_a)
        if (self.alpha(time)**2)*self.hs < self.threshold and self.state == 1:
           discharge = 0.
           
        self.setNew(cp - b*discharge, cm + b*discharge, discharge)

        


class Tailwater(Node):
    def __init__(self, x, read_tailwater_data):
        super().__init__(x or 0.)
        self.water_level = float(read_tailwater_data['water_level'])
        self.water_level = self.water_level  # alias para compatibilidad con steady_state
        self.setInit(self.water_level, 0.0)
 
    def solveAsDbc(self, b, cp, time):
        discharge = (cp - self.water_level) / b
        if discharge < 0.:      # válvula de retención: no permite flujo inverso
            discharge = 0.
            head = cp           # la cabeza la impone la característica positiva
        else:
            head = self.water_level
        self.setNew(head, discharge)

    def apply_dbc(self, mesh, time):
        node = mesh[-2]
        cp = node.old_head[0] + node.old_discharge[0] * (mesh.b - mesh.r * abs(node.old_discharge[0]))
        discharge = (cp - self.water_level) / mesh.b
        return self.water_level, discharge
 
    def steady_state_term(self, discharge_0):
        return -self.water_level

    def diff_steady_state_term(self, discharge_0):
        return 0.

# class DeviceInline:
#     def __init__(self, read_device_data, pipe, input_data):
#         # Asegurate de que la key coincida con cómo lee "surge_tank" el lector de inputs
#         if 'surge_tank' in read_device_data:
#             # Pasa los datos específicos del SURGE_TANK
#             self.device = SurgeTank(read_device_data['surge_tank'], input_data.surge_tank_data)
#         elif 'valve_inline' in read_device_data:
#             self.device = Valve(read_device_data['valve_inline'], input_data.valve_inline_data)
    
#         if read_device_data != {}:
#             self.position = float(read_device_data['position'])
#             if 'valve' in read_device_data.keys():
#                 valve_inline_data = read_device_data['valve']
#                 self.device = Valve(None, None, None, pipe, valve_inline_data)
#             elif 'surge_tank' in read_device_data.keys():
#                 surge_tank_data = read_device_data['surge_tank']
#                 self.device = SurgeTank(None, surge_tank_data)
#             elif 'pump' in read_device_data.keys():
#                 pump_inline_data = read_device_data['pump']
#                 modo = int(pump_inline_data['startup-shutdown'])
#                 estado_bomba = 0 if modo == 2 else 1
#                 self.device = Pump(None, None, None, pipe, pump_inline_data, state=estado_bomba)
#             if self.device is not None:
#                 self.device.x = self.position
#             elif 'air_chamber' in read_device_data.keys():
#                 air_chamber_data = read_device_data['air_chamber']
#                 self.device = AirChamber(None, None, None, pipe, air_chamber_data)
#             else:
#                 raise Exception('There was a problem while reading the DEVICES block')  #Quizá esto sea redundante

class DeviceInline:
    def __init__(self, read_device_data, pipe, input_data=None):
        self.device = None
        
        # 1. Si no hay dispositivos cargados, salimos
        if not read_device_data:
            return
            
        # 2. Obtenemos la posición de la clave correcta
        self.position = float(read_device_data['position'])

        # 3. Identificamos qué dispositivo es, y le pasamos SUS propios datos
        if 'surge_tank' in read_device_data:
            # Fíjate que acá le pasamos read_device_data['surge_tank'] que es el diccionario con los datos
            self.device = SurgeTank(read_device_data['surge_tank'], self.position)
            
        elif 'valve_inline' in read_device_data:
            self.device = Valve(None, None, None, pipe, read_device_data['valve_inline'])
            
        elif 'valve' in read_device_data:
            self.device = Valve(None, None, None, pipe, read_device_data['valve'])
            
        elif 'pump' in read_device_data.keys():
            pump_inline_data = read_device_data['pump']
            self.device = Pump(None, None, None, pipe, pump_inline_data)
            # state se define automáticamente en Pump.__init__ según mode
            
        elif 'air_chamber' in read_device_data:
            self.device = AirChamber(None, None, None, pipe, read_device_data['air_chamber'])
            
        # 4. Forzamos la asignación de la coordenada x
        if self.device is not None:
            self.device.x = self.position

class SurgeTank(Node):
    def __init__(self, data, x):
        super().__init__(x)
        self.diameter = float(data['diameter'])
        self.friction_coefficient = float(data['friction_coefficient'])
        self.cdap = float(data['loss_coefficient_for_positive_flow'])
        self.cdar = float(data['loss_coefficient_for_reverse_flow'])
        self.area = (math.pi/4.)*self.diameter**2
        self.old_time = 0.
        self.depths = []
        self.st_heads = []
        self.st_discharges = []

    def __repr__(self):
        eps = 0.001
        return (f'{self.x - eps:10.4f} {self.old_head[0]:10.4f} {self.old_discharge[0]:10.4f}\n'
                f'{self.x + eps:10.4f} {self.old_head[0]:10.4f} {self.old_discharge[1]:10.4f}')

    def setInit(self, head, discharge):
        self.setOld([head], [discharge, discharge])
        self.st_discharges.append(0.)
        self.depths.append(head)
        self.st_heads.append(head)

    def setNew(self, head, discharge_upstream, discharge_downstream):
        self.new_head = [head]
        self.new_discharge = [discharge_upstream, discharge_downstream]

    def steady_state_term(self, discharge):
        return 0.

    def diff_steady_state_term(self, discharge):
        return 0.

    def solve(self, b, cp, cm, time):
        delta_time = time - self.old_time
        self.old_time = time
        delta_depth = 0.5*(delta_time/self.area)*self.st_discharges[-1]
        
        if 0.5*(cp + cm) + self.st_heads[-1] - 2.*self.depths[-1] + (0.5*(delta_time/self.area) + 2.*self.depths[-1]/(Const.G*self.area*delta_time))*self.st_discharges[-1] - 1./(Const.G*self.area**2)*(self.friction_coefficient*self.depths[-1]/self.diameter)*self.st_discharges[-1]*abs(self.st_discharges[-1]) > 0.:
            #Flow towards the surge tank
            coeff_a = -1./(2.*Const.G*self.cdap**2)
            coeff_b = -0.5*b - 0.5*delta_time/self.area - 2.*self.depths[-1]/(Const.G*self.area*delta_time)
            coeff_c = 0.5*(cp + cm) + self.st_heads[-1] - 2.*self.depths[-1] + (0.5*(delta_time/self.area) + 2.*self.depths[-1]/(Const.G*self.area*delta_time))*self.st_discharges[-1] - 1./(Const.G*self.area**2)*(self.friction_coefficient*self.depths[-1]/self.diameter)*self.st_discharges[-1]*abs(self.st_discharges[-1])
            self.st_discharges.append((-coeff_b - math.sqrt(coeff_b**2 - 4.*coeff_a*coeff_c))/(2.*coeff_a))
            head = 0.5*(cp + cm - b*self.st_discharges[-1])
            self.st_heads.append(head - ((self.st_discharges[-1]/self.cdap)**2)/(2.*Const.G))
        else:
            #Flow from the surge tank
            coeff_a = 1./(2.*Const.G*self.cdar**2)
            coeff_b = -0.5*b - 0.5*delta_time/self.area - 2.*self.depths[-1]/(Const.G*self.area*delta_time)
            coeff_c = 0.5*(cp + cm) + self.st_heads[-1] - 2.*self.depths[-1] + (0.5*(delta_time/self.area) + 2.*self.depths[-1]/(Const.G*self.area*delta_time))*self.st_discharges[-1] - 1./(Const.G*self.area**2)*(self.friction_coefficient*self.depths[-1]/self.diameter)*self.st_discharges[-1]*abs(self.st_discharges[-1])
            self.st_discharges.append((-coeff_b - math.sqrt(coeff_b**2 - 4.*coeff_a*coeff_c))/(2.*coeff_a))
            head = 0.5*(cp + cm - b*self.st_discharges[-1])
            self.st_heads.append(head + ((self.st_discharges[-1]/self.cdar)**2)/(2.*Const.G))

        self.depths.append(0.5*(delta_time/self.area)*self.st_discharges[-1] + delta_depth + self.depths[-1])
        discharge_upstream = (cp - head)/b
        discharge_downstream = discharge_upstream - self.st_discharges[-1]
        self.setNew(head, discharge_upstream, discharge_downstream)


class AirChamber(Node):
    def __init__(self, x, head, discharge, read_air_chamber_data):
        super().__init__(x or 0.)
        
        self.volume = float(read_air_chamber_data['volume'])
        self.length = float(read_air_chamber_data['throat_length'])
        self.diameter = float(read_air_chamber_data['throat_diameter'])
        self.friction_coefficient = float(read_air_chamber_data['throat_friction_coefficient'])
        self.area = (math.pi/4.)*self.diameter**2
        self.constant = Const.Hatm*self.volume**Const.N

        self.downstream_head = None
        self.downstream_discharge = None

    def steady_state_term(self, discharge_0):
        return 0.

    def diff_steady_state_term(self, discharge_0):
        return 0.

    def solve(self, mesh, cp, cm, time):
        head = 0.5*(cp + cm)
        discharge_upstream = (cp - head)/mesh.b
	
        dummy_discharge = self.old_discharge[0]
        for i in range(Const.maxIter):
            H_ac_new = self.constant/(self.volume - 0.5*(dummy_discharge - self.old_discharge[0])*mesh.delta_time)**Const.N - Const.Hatm  # N = polytropic index
            c2 = 2.*self.length/(Const.G*self.area*mesh.delta_time)
            c1 = self.old_head[0] - head + self.friction_coefficient*self.length*self.old_discharge[0]*abs(self.old_discharge[0])/(Const.G*self.diameter*self.area**2) - c2*self.old_discharge[0]
            Q_ac_new = (H_ac_new - head - c1)/c2

            if abs(Q_ac_new - dummy_discharge) < Const.tolerance:
                break
            dummy_discharge = Q_ac_new

        discharge_downstream = discharge_upstream - Q_ac_new

        # Usamos setNew para integrarse con update() -> setOld
        self.setNew(head, discharge_upstream)          # new_head=[head], new_discharge=[discharge_upstream]
        self.new_discharge = [discharge_upstream, discharge_downstream]  # sobreescribir con ambos

        self.downstream_head = head
        self.downstream_discharge = discharge_downstream

        return head, discharge_upstream

#####################################################

class Turbine(Node):
    def __init__(self, x, head, discharge, read_turbine_data):
        super().__init__(x)
        self.nominal_head = float(read_turbine_data['nominal_head'])
        self.nominal_discharge = float(read_turbine_data['nominal_discharge'])
        self.nominal_speed = float(read_turbine_data['nominal_speed'])
        self.nominal_torque = float(read_turbine_data['nominal_torque'])
        self.initial_power = float(read_turbine_data['initial_power'])
        self.current_power = self.initial_power
        self.past_power = self.current_power
        self.delta_power = float(read_turbine_data['power_variation'])
        self.start_time = float(read_turbine_data['start_time'])
        self.end_time = float(read_turbine_data['end_time'])
        self.weight = float(read_turbine_data['weight'])
        self.gyration_radius = float(read_turbine_data['gyration_radius'])
        self.td = float(read_turbine_data['dashpot_time_constant'])
        self.tal = float(read_turbine_data['speed_servomotor_ratio'])
        self.sigma = float(read_turbine_data['sigma'])
        self.delta = float(read_turbine_data['delta'])
        self.turbine_datafile = read_turbine_data['dimensionless_data']
        self.zero_theta_line = int(read_turbine_data['zero_theta_line'])
        self.dtheta = float(read_turbine_data['dtheta'])*math.pi/180
        data = open(self.turbine_datafile, 'r').read()
        self.dy = float(read_turbine_data['dy'])
        blocks = data.split('\n\n')
        self.wh = [[float(element) for element in line.split()] for line in blocks[0].split('\n') if line.strip()]
        self.wb = [[float(element) for element in line.split()] for line in blocks[1].split('\n') if line.strip()]
        self.c32 = 3.00E+07/(math.pi*self.nominal_speed*self.nominal_torque)
        self.tdta = self.td*self.tal
        self.talp = self.tal + self.delta*self.td
        self.tm = self.weight*(self.gyration_radius**2)*self.nominal_speed*2.*math.pi/(60.*Const.G*self.nominal_torque)
        self.hill_chart_terms = 4*[None]
        self.nu = None
        self.alpha = 1.
        self.alpha0 = 1.
        self.beta = None
        self.theta = None
        self.y = None
        self.y0 = None
        self.z0 = 0.
        self.y_array = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
        self.file = open('salida.txt', 'w')

    def __repr__(self):
        return (f'{self.x:10.4f} {self.old_head[0]:10.4f} {self.old_discharge[0]:10.4f}\n'
                f'{self.x:10.4f} {self.old_head[1]:10.4f} {self.old_discharge[0]:10.4f}')

    def setInit(self, head_upstream, discharge):
        self.setOld([head_upstream, 0.0], [discharge])

    def setNew(self, head_upstream, head_downstream, discharge):
        self.new_head = [head_upstream, head_downstream]
        self.new_discharge = [discharge]

    def steady_state_term(self, discharge_0):
        self.beta = self.c32*self.initial_power
        self.nu = discharge_0/self.nominal_discharge
        wbb = self.beta/(1. + self.nu**2)
        self.theta = math.atan2(self.nu, 1.0)
        i = math.floor(self.theta/self.dtheta) + self.zero_theta_line
        xx = (self.theta - (i - self.zero_theta_line)*self.dtheta)/self.dtheta

        for j in range(len(self.wb[0])):
            c1 = (1. - xx)*self.wb[i][j] + xx*self.wb[i + 1][j]
            if c1 > wbb:
                break

        if j == 0:
            self.y = 1.
            wha = self.wh[i][j]
            whb = self.wh[i + 1][j]
        else:
            c2 = (1. - xx)*self.wb[i][j - 1] + xx*self.wb[i + 1][j - 1]
            cj = (wbb - c2)/(c1 - c2)
            self.y = 1. - (j - 1)*self.dy - cj*self.dy
            wha = self.wh[i][j - 1]*(1. - cj) + self.wh[i][j]*cj
            whb = self.wh[i + 1][j - 1]*(1. - cj) + self.wh[i + 1][j]*cj

        self.hill_chart_terms[0] = (whb - wha)/self.dtheta
        self.hill_chart_terms[1] = wha - self.hill_chart_terms[0]*(i - self.zero_theta_line)*self.dtheta

        # Calcular wb_val para inicializar beta correctamente desde el hill chart
        if j == 0:
            wba = self.wb[i][j]
            wbb_row = self.wb[i + 1][j]
        else:
            wba = self.wb[i][j - 1]*(1. - cj) + self.wb[i][j]*cj
            wbb_row = self.wb[i + 1][j - 1]*(1. - cj) + self.wb[i + 1][j]*cj
        b1 = (wbb_row - wba)/self.dtheta
        b0 = wba - b1*(i - self.zero_theta_line)*self.dtheta
        wb_val = b0 + b1*self.theta
        self.beta = (1. + self.nu**2)*wb_val

        # Guardar estado para el primer paso transitorio
        self.alpha0 = 1.
        self.y0 = self.y


        return -(self.hill_chart_terms[1] + self.hill_chart_terms[0]*self.theta)*(1. + self.nu**2)*self.nominal_head

    def diff_steady_state_term(self, discharge_0):
        return -self.nominal_head*(2.*self.nu*(self.hill_chart_terms[1] + self.hill_chart_terms[0]*self.theta) + self.hill_chart_terms[0])/self.nominal_discharge

    def solveAsDbc(self, mesh, cp, time):

        if time <= self.start_time:
            self.setNew(self.old_head[0], self.old_head[1], self.old_discharge[0])
            return

        elif time > self.start_time and time <= self.end_time:
            self.past_power = self.current_power
            self.current_power = self.initial_power + (time - self.start_time)*self.delta_power/(self.end_time - self.start_time)
        else:
            self.past_power = self.current_power
            self.current_power = self.initial_power + self.delta_power

        alpha0 = self.alpha0
        y0 = self.y0

        for n in range(Const.maxIter):
            self.theta = math.atan2(self.nu, self.alpha)
            i = math.floor(self.theta/self.dtheta) + self.zero_theta_line

            if self.y >= self.y_array[0]:
                wha = self.wh[i][0]
                whb = self.wh[i + 1][0]
                wba = self.wb[i][0]
                wbb = self.wb[i + 1][0]
            elif self.y <= self.y_array[-1]:
                wha = self.wh[i][-1]
                whb = self.wh[i + 1][-1]
                wba = self.wb[i][-1]
                wbb = self.wb[i + 1][-1]
            else:
                for k in range(len(self.y_array) - 1):
                    if self.y < self.y_array[k] and self.y > self.y_array[k + 1]:
                        ck = (self.y_array[k] - self.y)/(self.y_array[k] - self.y_array[k + 1])
                        wha = self.wh[i][k]*(1. - ck) + self.wh[i][k + 1]*ck
                        whb = self.wh[i + 1][k]*(1. - ck) + self.wh[i + 1][k + 1]*ck
                        wba = self.wb[i][k]*(1. - ck) + self.wb[i][k + 1]*ck
                        wbb = self.wb[i + 1][k]*(1. - ck) + self.wb[i + 1][k + 1]*ck
                        break

            a1 = (whb - wha)/self.dtheta
            a0 = wha - a1*(i - self.zero_theta_line)*self.dtheta
            b1 = (wbb - wba)/self.dtheta
            b0 = wba - b1*(i - self.zero_theta_line)*self.dtheta
            self.hill_chart_terms = [a1, a0, b1, b0]

            wh_val = a0 + a1*self.theta
            wb_val = b0 + b1*self.theta
            c8 = self.alpha**2 + self.nu**2

            f1 = cp - mesh.b*self.nu*self.nominal_discharge - self.nominal_head*c8*wh_val
            f2 = c8*wb_val + self.beta - self.c32*(self.current_power/self.alpha + self.past_power/alpha0) - 2.*self.tm*(self.alpha - alpha0)/mesh.delta_time

            df1_dal = -self.nominal_head*wh_val*2.*self.alpha + self.nominal_head*a1*self.nu
            df1_dnu = -self.nominal_discharge*mesh.b - self.nominal_head*wh_val*2.*self.nu - self.nominal_head*a1*self.alpha
            df2_dal = 2.*self.alpha*wb_val - b1*self.nu + self.c32*self.current_power/self.alpha**2 - 2.*self.tm/mesh.delta_time
            df2_dnu = 2.*self.nu*wb_val + b1*self.alpha

            if self.y != 0.:
                dz_dt = -(self.y - y0)*self.talp/(mesh.delta_time*self.tdta) - (self.sigma*(0.5*(self.y + y0) - 1.) + 0.5*(self.alpha + alpha0) - 1.)/self.tdta - (self.alpha - alpha0)/(mesh.delta_time*self.tal)
                f3 = 2.*((self.y - y0)/mesh.delta_time - self.z0) - dz_dt*mesh.delta_time
                df3_dal = 0.5*mesh.delta_time/self.tdta + 1./self.tal
                df3_dy = (self.talp + 0.5*mesh.delta_time*self.sigma)/self.tdta + 2./mesh.delta_time

                dete = df1_dal*df2_dnu*df3_dy - df1_dnu*df2_dal*df3_dy
                delta_y = (-f3*df1_dal*df2_dnu - f2*df3_dal*df1_dnu + f1*df2_dnu*df3_dal + f3*df1_dnu*df2_dal)/dete
                self.y += delta_y
                if self.y > 1.:
                    self.y = 1.

            df3_dy = 1.
            dete = df2_dnu*df1_dal - df1_dnu*df2_dal
            delta_alpha = (-f1*df2_dnu + f2*df1_dnu)/dete
            delta_nu = (f1*df2_dal - f2*df1_dal)/dete

            self.alpha += delta_alpha
            self.nu += delta_nu

            if abs(delta_alpha) + abs(delta_nu) < Const.tolerance:
                break

        self.beta = c8*wb_val
        self.z0 = 2.*(self.y - y0)/mesh.delta_time - self.z0
        self.alpha0 = self.alpha
        self.y0 = self.y

        self.file.write('{:10.4f} {:10.4f} {:10.4f} {:10.4f} {:10.4f}\n'.format(
            time, self.nu, self.alpha, self.y, self.beta))

        discharge = self.nu*self.nominal_discharge
        head = cp - mesh.b*discharge
        self.setNew(head, 0., discharge)
        
########################################################################################################

class Turbopump(Node):
    def __init__(self, x, head, discharge, read_turbopump_data):
        super().__init__(x = 0.)
        self.nominal_head = float(read_turbopump_data['nominal_head'])								#Hr
        self.nominal_discharge = float(read_turbopump_data['nominal_discharge'])					#Qr
        self.nominal_speed = float(read_turbopump_data['nominal_speed'])							#Nr
        self.nominal_torque = float(read_turbopump_data['nominal_torque'])							#Tr
        self.weight = float(read_turbopump_data['weight'])											#weight
        self.gyration_radius = float(read_turbopump_data['gyration_radius'])						#radius
        self.turbopump_datafile = read_turbopump_data['dimensionless_data']
        self.dtheta = float(read_turbopump_data['dtheta'])
        data = open(self.turbopump_datafile, 'r').read()
        self.wh = [[float(e) for e in line.split()] for line in data.split('\n\n')[1].split('\n')]
        self.wb = [[float(element) for element in line.split()] for line in data.split('\n\n')[2].split('\n')]		
        #En lista de listas wh[i][j], i refiere a la fila y j refiere a la columna. len(wh[0]) = 11, lo mismo que len(y_array)
        self.hill_chart_terms = 4*[None]
        self.nu = None
        self.alpha = 1.
        self.beta = None
        self.theta = None
        self.c31 = math.pi*self.weight*(self.gyration_radius**2)*self.nominal_speed/(15.*Const.G*self.nominal_torque)   #/mesh.delta_time
		
    def steady_state_term(self, discharge_0):
        self.nu = discharge_0/self.nominal_discharge
        self.theta = math.pi + math.atan(self.nu)                                                   #Caso especial de la fórmula general theta = pi + atan(v/al)
        i = math.floor(self.theta/self.dtheta)

        wha = self.wh[i][0]                                                                           #Este arreglo es de dos dimensiones. La segunda dimensión corresponde a la velocidad angular nominal
        whb = self.wh[i + 1][0]

        self.hill_chart_terms[0] = (whb - wha)/self.dtheta
        self.hill_chart_terms[1] = wha - self.hill_chart_terms[0]*i*self.dtheta

        return (self.hill_chart_terms[1] + self.hill_chart_terms[0]*self.theta)*(1. + self.nu**2)*self.nominal_head

    def diff_steady_state_term(self, discharge_0):
        return (2.*self.nu*(self.hill_chart_terms[1] + self.hill_chart_terms[0]*self.theta) + self.hill_chart_terms[0])*self.nominal_head/self.nominal_discharge

####################################################

    def solve(self, mesh, cp, cm, time):

        solution = fsolve(self.function, [self.alpha, self.nu], args = (mesh, cp, cm), fprime = self.jacobian, xtol = Const.tolerance)
        discharge = solution[1]*self.nominal_discharge
        self.downstream_discharge = discharge
        head = cp - mesh.b*discharge
        self.downstream_head = head + (self.hill_chart_terms[1] + self.hill_chart_terms[0]*self.theta)*(solution[0]**2 + solution[1]**2)*self.nominal_head

        return head, discharge
		
####################################################

    def function(self, x, mesh, cp, cm):

        self.theta = math.pi + math.atan(self.nu/self.alpha)
        i = math.floor(self.theta/self.dtheta)

        wha = self.wh[i][0]
        whb = self.wh[i + 1][0]
        wba = self.wb[i][0]
        wbb = self.wb[i + 1][0]
    
        self.hill_chart_terms[0] = (whb - wha)/self.dtheta							#a1
        self.hill_chart_terms[1] = wha - self.hill_chart_terms[0]*i*self.dtheta		#a0
    
        self.hill_chart_terms[2] = (wbb - wba)/self.dtheta							#b1
        self.hill_chart_terms[3] = wba - self.hill_chart_terms[2]*i*self.dtheta		#b0
        
        return np.array([cp - cm - 2.*mesh.b*x[1]*self.nominal_discharge + self.nominal_head*(x[0]**2 + x[1]**2)*(self.hill_chart_terms[1] + self.hill_chart_terms[0]*self.theta),
                         (x[0]**2 + x[1]**2)*(self.hill_chart_terms[3] + self.hill_chart_terms[2]*self.theta) + self.beta - self.c31*(self.alpha - x[0])/mesh.delta_time])

####################################################

    def jacobian(self, x, mesh, cp):
        return np.array([[self.nominal_head*(2.*x[0]*(self.hill_chart_terms[1] + self.hill_chart_terms[0]*self.theta) - self.hill_chart_terms[0]*x[1]),	
                          -2.*mesh.b*self.nominal_discharge + self.nominal_head*(2.*x[1]*(self.hill_chart_terms[1] + self.hill_chart_terms[0]*self.theta) + self.hill_chart_terms[0]*x[0])],
                         [2.*x[0]*(self.hill_chart_terms[3] + self.hill_chart_terms[2]*self.theta) - x[1]*self.hill_chart_terms[2] + self.c31/mesh.delta_time,
                          2.*x[1]*(self.hill_chart_terms[3] + self.hill_chart_terms[2]*self.theta) + x[0]*self.hill_chart_terms[2]]])

########################################################################################################



class Output:
    def __init__(self, read_output_data):
        required_data = ['output_file', 'nbr_of_els', 'time_step', 'output_frequency', 'simulation_time']
        for item in required_data:
            if item not in read_output_data.keys():
                raise Exception('There was a problem while reading the OUTPUT block')
        self.file_name = read_output_data['output_file']
        self.nbr_of_els = int(read_output_data['nbr_of_els'])
        self.time_step = float(read_output_data['time_step'])
        self.output_frequency = int(read_output_data['output_frequency'])
        self.simulation_time = float(read_output_data['simulation_time'])
        self.file = open(self.file_name, 'w')
        self.head_lims = [float('inf'), float('-inf')]
        self.discharge_lims = [float('inf'), float('-inf')]  
        # self.head_lims = [0., 0.]
        # self.discharge_lims = [0., 0.]
	
    def write_file(self, mesh):
        for node in mesh:
            # Si el nodo tiene dos cabezas (bomba), escribir dos líneas con
            # posiciones x ligeramente distintas para visualizar el salto de
            # presión sin generar el pico visual artefacto (línea vertical).
            # Se usa x-epsilon para H_up (aguas arriba) y x+epsilon para H_down
            # (aguas abajo), donde epsilon es un valor muy pequeño respecto a
            # la longitud de la tubería.
            if hasattr(node, 'new_head') and node.old_head is not None and len(node.old_head) == 2:
                eps = mesh.delta_x * 0.01
                self.file.write(
                    f"{node.x - eps} {node.old_head[0]} {node.old_discharge[0]}\n")
                self.file.write(
                    f"{node.x + eps} {node.old_head[1]} {node.old_discharge[0]}\n")
            else:
                self.file.write(
                    f"{node.x} {node.old_head[0]} {node.old_discharge[0]}\n")
        self.file.write('\n')
	
    def update_plot_lims(self, head, discharge):
        self.head_lims[0] = min(self.head_lims[0], min(head))
        self.head_lims[1] = max(self.head_lims[1], max(head))
        self.discharge_lims[0] = min(self.discharge_lims[0], min(discharge))
        self.discharge_lims[1] = max(self.discharge_lims[1], max(discharge))
    # def update_plot_lims(self, head, discharge):
    #     if self.head_lims[0] > min(head):
    #         self.head_lims[0] = min(head)
    #     if self.head_lims[1] < max(head):
    #         self.head_lims[1] = max(head)
    #     if self.discharge_lims[0] > min(discharge):
    #         self.discharge_lims[0] = min(discharge)    
    #     if self.discharge_lims[1] < max(discharge):
    #         self.discharge_lims[1] = max(discharge)
    def close(self):
        self.file.close()

def Viz(filename, videofile, x_max, delta_time, xlabel, ylabel, y2label, dpi_value):
    fig = plt.figure()
    fig.set_dpi(dpi_value)
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    data = open(filename, 'r').read()
    data_blocks = data.split('\n\n')
    nbr_of_frames = len(data_blocks) - 1
    line01, = ax1.plot([], [], lw = 2, color = 'r')
    line02, = ax2.plot([], [], lw = 2, color = 'b')
    time_text = ax1.text(0.1, 0.1, "", transform = ax1.transAxes, fontsize = 15)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_xlim(0., x_max)
    ax1.set_ylim(-10., 210.)
            
    ax2.set_ylabel(y2label)
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0., 1.0)
    
    def init():
        line01.set_data([], [])    
        line02.set_data([], [])
        return line01, line02
    
    def animate(i):
        x, h, q = [], [], []
        block = data_blocks[i].strip() 
        for line in block.split('\n'):
            res = line.split()
            if len(res) == 3:
                # Corrección: acceder a cada elemento de la lista por índice
                x_val = float(res[0].replace('[','').replace(']',''))
                h_val = float(res[1].replace('[','').replace(']',''))
                q_val = float(res[2].replace('[','').replace(']',''))
                x.append(x_val)
                h.append(h_val)
                q.append(q_val)
        
        line01.set_data(x, h)
        line02.set_data(x, q)
        time = float(i) * delta_time
        time_text.set_text("Time: %.2f s" % time)
        return line01, line02, time_text
    
    anim = animation.FuncAnimation(fig, animate, init_func = init,
    frames = nbr_of_frames, interval = 100, blit = True)
    anim.save(videofile, writer = "ffmpeg")