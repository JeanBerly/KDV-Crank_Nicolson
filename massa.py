import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.rcParams['axes.formatter.useoffset'] = False
mass = []
x = []
x = np.linspace(0, 20, 500)
m = open('massa.txt', 'r')
mass_data = m.readlines()[0:500]
for i in range(500):
    mass.append(float(mass_data[i]))

plt.title("Variação da massa:")
plt.xlabel("tempo (s)")
plt.ylabel("massa")
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.7f'))
# plt.xlim(-100, 0)
# plt.ylim(-5, 15)
plt.plot(x, mass)
plt.savefig("massa.png")
plt.show()
m.close()
