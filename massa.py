import numpy as np
import matplotlib.pyplot as plt

mass = []
x = []
x = np.linspace(-100, 0, 1000)
m = open('mass_data.txt', 'r')
mass_data = m.readlines()[0:1000]
for i in range(1000):
    mass.append(float(mass_data[i]))

plt.title("Variação da massa:")
plt.xlabel("x")
plt.ylabel("massa")
# plt.xlim(-100, 0)
# plt.ylim(-5, 15)
plt.plot(x, mass)
plt.show()
m.close()
