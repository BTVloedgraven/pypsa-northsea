#Small network test non-linear costs

import pypsa
import numpy as np
import math
n = pypsa.Network()

n_OWF = 3
d_OWF_hub = 25
d_hub_load = 150
ang_offset = np.pi/2 #First OWF North of hub

load = 2000 #MW

hub_loc = (0,0)
n.add("Bus", "Hub bus", x=hub_loc[0], y=hub_loc[1])

load_loc = (hub_loc[0]+d_hub_load,hub_loc[1])
n.add("Bus", "Load bus", x=load_loc[0], y=load_loc[1])
n.add("Link", "DC hub-load", bus0="Hub bus", bus1="Load bus", length=d_hub_load, p_nom_extendable=True)
n.add("Load", "Load", bus="Load bus", p_set=load)

OWF_loc = []
for i in range(n_OWF):
    ang = i*2*np.pi/n_OWF + ang_offset
    OWF_loc.append((d_OWF_hub*math.cos(ang), (d_OWF_hub*math.sin(ang))))
    n.add("Bus", f"OWF{i} bus", x=OWF_loc[i][0], y=OWF_loc[i][1])
    n.add("Generator", f"OWF{i}", bus=f"OWF{i} bus", p_nom=load/n_OWF)
    d_OWF_load = np.sqrt((load_loc[0]-OWF_loc[i][0])**2+(load_loc[1]-OWF_loc[i][1])**2)
    n.add("Link", f"DC OWF{i}-load", bus0=f"OWF{i} bus", bus1="Load bus", length=d_OWF_load, p_nom_extendable=True)
    n.add("Line", f"AC OWF{i}-hub", bus0=f"OWF{i} bus", bus1="Hub bus", length=d_OWF_hub, s_nom_extendable=True)

ac_station_cost = 250e3 #EUR/MW
dc_station_cost = 400e3 #EUR/MW

ac_cable_cost = 2600 #EUR/MW/km
dc_cable_cost = 1300 #EUR/MW/km

n.lines.capital_cost = n.lines.apply(lambda line: line.length*ac_cable_cost + ac_station_cost, axis=1)
n.links.capital_cost = n.links.apply(lambda link: link.length*dc_cable_cost + dc_station_cost, axis=1)

n.optimize(solver_name="gurobi")