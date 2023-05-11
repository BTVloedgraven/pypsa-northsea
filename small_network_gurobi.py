

import gurobipy as gp
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# ac_station_cost = 250e3
dc_station_cost = 600e3 #EUR/MW

ac_cable_cost = 500e3/90 #EUR/MW/km to hub
dc_cable_cost = 1300 #EUR/MW/km

#non-linear costs
# P = [0, 500, 1000, 2000, 4000]
# dc_station_cost_nonlin = [0, 437e6, 591e6, 800e6, 1600e6]
# dc_cable_cost_nonlin = [0, 1.4e6, 1.9e6, 2.7e6, 5.4e6]

GBPtoEUR2016 = 1.4

e_cable = 0.44
e_station = 0.65
P_ref = 1200
c_cable_ref = 1.1e6*GBPtoEUR2016
c_station_ref = 630e6*GBPtoEUR2016
a = c_cable_ref/P_ref**e_cable
b = c_station_ref/P_ref**e_station

P = [0,500,1200,2000]
dc_cable_cost_nonlin = [a*p**e_cable for p in P]
dc_station_cost_nonlin = [b*p**e_station for p in P]
# dc_station_cost_nonlin = [0, 554e6, 815e6, 1200e6, 2400e6]
# dc_cable_cost_nonlin = [0, 1.375e6, 1.925e6, 2.700e6, 5.400e6]

fig, ax = plt.subplots(figsize=(4,5))

ax.plot([P[i] for i in (0, -1)], [0,c_station_ref/P_ref*P[-1]], 'o--')
ax.plot(P, dc_station_cost_nonlin, 'o-')

current_values = ax.get_yticks()
ax.set_yticklabels(['{:.1f}'.format(x*1e-6) for x in current_values])

ax.set_xlabel('Station capacity [MW]')
ax.set_ylabel('Capex [M€]')

fig.tight_layout()
fig.savefig('visuals/dc_station_PWL.png')

fig, ax = plt.subplots(figsize=(4,5))

ax.plot([P[i] for i in (0, -1)], [0,c_cable_ref/P_ref*P[-1]], 'o--')
ax.plot(P, dc_cable_cost_nonlin, 'o-')

current_values = ax.get_yticks()
ax.set_yticklabels(['{:.1f}'.format(x*1e-6) for x in current_values])

ax.set_xlabel('Transmission capacity [MW]')
ax.set_ylabel('Capex [M€/km]')

fig.tight_layout()
fig.savefig('visuals/dc_cable_PWL.png')

def small_network(n_OWF=3, cap_per_OWF=1000, d_OWF_hub=20, d_hub_load=150, ang_offset=np.pi/4, non_lin=False, hub=True, radial=True):
    hub_loc = (0,0)
    load_loc = (hub_loc[0]+d_hub_load,hub_loc[1])

    m = gp.Model("Small network")

    power = {}
    ac_lines = {}
    dc_lines = {}
    OWF_locs = {}
    d_OWF_load = {}

    for i in range(n_OWF):
        ang = i*2*np.pi/n_OWF + ang_offset
        power[f"OWF{i}"] = cap_per_OWF
        OWF_locs[f"OWF{i}"] = (d_OWF_hub*math.cos(ang), (d_OWF_hub*math.sin(ang)))
        d_OWF_load[f"OWF{i}-load"] = np.sqrt((load_loc[0]-OWF_locs[f"OWF{i}"][0])**2+(load_loc[1]-OWF_locs[f"OWF{i}"][1])**2)

        ac_lines[f"OWF{i}-hub"] = m.addVar(name=f"AC OWF{i}-hub", obj=ac_cable_cost*d_OWF_hub)
        dc_lines[f"OWF{i}-load"] = m.addVar(name=f"DC OWF{i}-load", obj=dc_station_cost+dc_cable_cost*d_OWF_load[f"OWF{i}-load"])

        m.addConstr(ac_lines[f"OWF{i}-hub"] + dc_lines[f"OWF{i}-load"] == power[f"OWF{i}"])

        if not hub:
            m.addConstr(ac_lines[f"OWF{i}-hub"] == 0)

        if not radial:
            m.addConstr(dc_lines[f"OWF{i}-load"] == 0)

    d_OWF_load["hub-load"] = d_hub_load
    dc_lines["hub-load"] = m.addVar(name=f"DC hub-load", obj=dc_station_cost+dc_cable_cost*d_hub_load)
    m.addConstr(gp.quicksum(ac_lines[f"OWF{i}-hub"] for i in range(n_OWF)) == dc_lines["hub-load"])
    if non_lin:
        for dc_line in dc_lines:
            m.setPWLObj(dc_lines[dc_line], P, np.add([dcs for dcs in dc_station_cost_nonlin], [cc*d_OWF_load[dc_line] for cc in dc_cable_cost_nonlin]))


    m.update()

    # Solve
    m.optimize()

    sol = pd.Series({v.VarName: v.x for v in m.getVars()})

    # if fig:
    #     bus_size_correction_factor = 5
    #     ax = plt.subplot()
    #     for OWF in OWF_locs:
    #         ax.add_patch(plt.Circle(OWF_locs[OWF], radius=np.sqrt(cap_per_OWF/np.pi)/bus_size_correction_factor))
    #     ax.add_patch(plt.Circle(load_loc, radius=np.sqrt(n_OWF*cap_per_OWF/np.pi)/bus_size_correction_factor, color='red'))
    #     ax.add_patch(plt.Circle(hub_loc, radius=np.sqrt(sol["DC hub-load"]/np.pi)/bus_size_correction_factor, color='green'))
    #     for dc_line in dc_lines:
    #         ax.plot()
    #     ax.axis('equal')
    #     ax.axis('off')

    #     plt.savefig(fig)

    return sol, m.ObjVal

n_OWF = 4
cap_per_OWF = 1000 #MW

output_lin, obj_lin = small_network(n_OWF=n_OWF, cap_per_OWF=cap_per_OWF, non_lin=False, hub=True, radial=True)
output_lin_hub, obj_lin_hub = small_network(n_OWF=n_OWF, cap_per_OWF=cap_per_OWF, non_lin=False, hub=True, radial=False)
output_nonlin, obj_nonlin = small_network(n_OWF=n_OWF, cap_per_OWF=cap_per_OWF, non_lin=True, hub=False, radial=True)
output_nonlin_hub, obj_nonlin_hub = small_network(n_OWF=n_OWF, cap_per_OWF=cap_per_OWF, non_lin=True, hub=True, radial=True)


