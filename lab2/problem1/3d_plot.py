import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import torch

grid_size = 100
model = torch.load('neural-network-1.pth').double()

state = lambda y,w: np.array([0,y,0,0,w,0,0,0])
w = np.linspace(-np.pi, np.pi, grid_size)
y = np.linspace(0, 1.5, grid_size)
P = np.nan*np.eye(grid_size)

for row in range(grid_size):
    for col in range(grid_size):
        s = torch.from_numpy(state(y[row],w[col]))
        P[row,col] = 1 + model(s.double()).argmax().item()
        # print(model(s.double()).max().item())
# print(y)
# print(w)
# print(P[0,:])
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
w,y = np.meshgrid(w,y)
surf = ax.plot_surface( y,w, P, 
                       linewidth=0, antialiased=False)

ax.set_xlabel('$y$')
ax.set_ylabel('$\omega$')
ax.set_zlabel(r'argmax$_aQ_{\theta}(s(y,\omega), a)$', rotation=60)
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('surface_policy.png')
plt.show()
