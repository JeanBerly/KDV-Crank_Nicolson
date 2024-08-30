import numpy as np
import matplotlib.pyplot as plt


x_axis = np.linspace(-100, 0, 2000)
u_axis = []
a = 0
b = 2000

aux = []

for i in range(1000):
    plt.title("Simulação na semirreta -  colisão com o limite do intervalo ")
    plt.xlabel("x")
    plt.ylabel("-u")
    plt.xlim(-100, 0)
    plt.ylim(-5, 20)
    f = open('kdv_data.txt', 'r')
    aux = f.readlines()[a:b]
    for j in range(2000):
        u_axis.append(float(aux[j]))

    plt.plot(x_axis, u_axis)
    plt.savefig(f'images_kdv/{i:00003}', dpi=100, facecolor='white')
    plt.close()
    u_axis.clear()
    a = a + 2001
    b = b + 2001
    f.close()

