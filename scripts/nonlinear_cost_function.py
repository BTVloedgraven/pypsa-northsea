import numpy as np
import matplotlib.pyplot as plt

inflation = 1.0519
a = inflation*0.125 + 0.1437
b = inflation*165

def C(P):
    return a*P + 2*b*np.arctan(P)/np.pi

def C2(P):
    return 20*P**0.5

P_values = [0, 0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100, 500, 1000, 2000, 3000, 5000]
fig, ax = plt.subplots()
ax.plot(P_values, [C(P) for P in P_values])
ax.plot(P_values, [C2(P) for P in P_values])
ax.plot(P_values, [0.4*P for P in P_values])
ax.set_xlabel('Platform capacity (MW)')
ax.set_ylabel('Cost (M\u20ac)')

