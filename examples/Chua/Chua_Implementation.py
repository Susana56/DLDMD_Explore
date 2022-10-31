import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# parameters
alpha = 15.395
beta = 28
R = -1.143
C_2 = -0.714

def chua(t, u):
    x, y, z = u
    # electrical response of the nonlinear resistor
    f_x = C_2*x + 0.5*(R-C_2)*(abs(x+1)-abs(x-1))
    dudt = [alpha*(y-x-f_x), x - y + z, -beta * y]
    return dudt


# time discretization
t_0 = 0
dt = 1e-2
t_final = 20
tspan = np.array([0, t_final])
t = np.arange(t_0, t_final, dt)

# initial conditions
u0 = [0.1, 0, 0]
# integrate ode system
sol = solve_ivp(chua, t_span=tspan, y0=u0, method='LSODA')

# 3d-plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.plot(sol.y[0, :],
        sol.y[1, :],
        sol.y[2, :])
ax.set_title("solve_ivp")

plt.show()
