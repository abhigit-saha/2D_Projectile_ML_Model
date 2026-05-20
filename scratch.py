import numpy as np
from models_3d import LinearProjectile3D

K = np.array([
    [2740.04, 0.0, 1685.87],
    [0.0, 3423.22, 2088.79],
    [0.0, 0.0, 1.0]
])

# Just mock some data based on what a trajectory might look like
t = np.linspace(0.5, 1.0, 14)
u = np.linspace(500, 1500, 14)
v = np.linspace(800, 200, 14) + 400 * (t - 0.75)**2 # Parabola opening upwards?
print(v)

model = LinearProjectile3D(K)
model.fit(t, u, v)

print("C parameters:", model.C)
