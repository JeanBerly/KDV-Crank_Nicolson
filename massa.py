import numpy as np
import matplotlib.pyplot as plt

mass = []
x = []
x = np.linspace(0, 40, 1000)
m = open('massa.txt', 'r')
mass_data = m.readlines()[0:1000]
for i in range(1000):
    mass.append(float(mass_data[i]))

plt.title("Variação da massa:")
plt.xlabel("tempo (s)")
plt.ylabel("massa")
# plt.xlim(-100, 0)
# plt.ylim(-5, 15)
plt.plot(x, mass)
plt.savefig("massa.png")
plt.show()
m.close()
