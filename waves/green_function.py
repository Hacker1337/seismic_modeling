import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


'''
В изотропной среде с скоростями продольных и поперечных волн vp и vs 
По заданному вектору смещения r = [x, y, z]
от источника гармонических колебаний с амплитудой a = [x, y, z]
программа считает результирующее смещение в точке смещенной от источника на r
в момент времени t
(это будет тоже вектор)
'''

vp, vs = 12, 10
rho = 1     # плотность среды
t = 100
w = 10      # частота волны
a = np.array([0, 1, 0])
# def f(t):
#     return a*np.cos(t*)
r = np.array([0, 30, 0])

# формула 4.10
# чтобы работала формула G*f, как ожидается,
def g4_10(R, a):
    R = np.array(R)
    kp = w/vp
    ks = w/vs
    r = norm(R)
    gamma = R / r
    result = np.dot(((gamma*np.expand_dims(gamma, 1)) / (4 * np.pi * rho * vp ** 2 * r) * np.exp(1j * kp * r) -
                     (gamma*np.expand_dims(gamma, 1) - np.eye(3)) / (4 * np.pi * rho * vs ** 2 * r) * np.exp(1j * ks * r)), a)
    return result

# print(np.abs(g4_10(r, a)))

plt.figure(figsize=(16, 12))
X, Y = np.linspace(-100, 100, 30), np.linspace(-100, 100, 30)
field = np.real(np.array([[g4_10([x, y, 0], a)[:2] if np.sqrt(x*x+y*y) > 10 else [0, 0] for x in X] for y in Y] ))
# plt.pcolormesh(np.sqrt(np.sum(field*field, -1)))
# plt.colorbar()
plt.quiver(*np.meshgrid(X, Y), field[:, :, 0], field[:, :, 1])
plt.show()


#  # на оси сигнал
# x = np.linspace(10, 100, 91)
# y = [np.real(g4_10([0, z, 0], a)[1]) for z in x]
# plt.plot(x, y)
# plt.show()