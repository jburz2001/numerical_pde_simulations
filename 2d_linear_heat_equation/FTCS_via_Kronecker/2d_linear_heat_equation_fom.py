# imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import h5py

# geometry and mesh
Lx, Ly = 1.0, 1.0
Nx, Ny = 30, 30
dx, dy = Lx / Nx, Ly / Ny
x      = np.linspace(0, Lx, Nx+1)
y      = np.linspace(0, Ly, Ny+1)
X, Y   = np.meshgrid(x, y, indexing='ij')

# physics
kappa  = 1.0e-2

# 1D Laplacian
def lap1d(n, h):
    main = -2.0 * np.ones(n-1)
    off  =  1.0 * np.ones(n-2)
    return sp.sparse.diags([off, main, off], offsets=[-1, 0, 1]) / h**2

# 1D Laplacians in x and y
Dx = lap1d(Nx, dx)
Dy = lap1d(Ny, dy)

# 2D Laplacian from Kronecker sum
Ix = sp.sparse.eye(Nx-1, format="csr")
Iy = sp.sparse.eye(Ny-1, format="csr")
D  = sp.sparse.kron(Iy, Lx) + sp.sparse.kron(Ly, Ix)

# timestepping
dt_max  = 1.0 / (2 * kappa * (1/dx**2 + 1/dy**2))
SF      = 0.25
dt      = SF * dt_max
t_final = 100.0
Nt      = int(t_final // dt)

# initial Gaussian pulse temperature field
mu_x, mu_y      = 0.5, 0.5
sigma_x, sigma_y = 0.15, 0.15
u_full = np.zeros((Nx + 1, Ny + 1))
gauss  = np.exp(
    -0.5 * ((X - mu_x) / sigma_x) ** 2
    -0.5 * ((Y - mu_y) / sigma_y) ** 2
)
u_full[1:-1, 1:-1] = gauss[1:-1, 1:-1]

# snapshots
# interior vector (Fortran order for compatability with Kronecker mathematics)
# operations are only performed on the interior points, so the Dirichlet BCs are implicitly applied
DOFs = (Nx+1)*(Ny+1)
Q = np.empty((DOFs, Nt + 1))
u = u_full[1:-1, 1:-1].ravel(order="F")
Q[:, 0] = u_full.ravel(order='F')
u = u.reshape((Nx-1, Ny-1), order="F")

# FTCS explicit scheme operator
A = sp.sparse.eye(D.shape[0], format="csr") + (kappa*dt)*D

# time marching loop
for k in range(1, Nt + 1):
    u = A @ u
    u_full[1:-1, 1:-1] = u.reshape((Nx-1, Ny-1), order="F")
    Q[:, k] = u_full.ravel(order='F')
    u = u.reshape((Nx-1, Ny-1), order="F")


# export snapshots
with h5py.File('2d_linear_heat_equation_fom.h5', 'w') as hf:
    hf.create_dataset('snapshots', data=Q)
    hf.create_dataset('domain_length_x', data=Lx)
    hf.create_dataset('domain_length_y', data=Ly)
    hf.create_dataset('timestep', data=dt)
    hf.create_dataset('number_of_timesteps', data=Nt)
    hf.create_dataset('final_time', data=t_final)
    hf.create_dataset('diffusion_constant', data=kappa)


# plot final temperature field
plt.figure()
plt.imshow(u_full.T, origin='lower', extent=[0, Lx, 0, Ly])
plt.colorbar(label='Temperature')
plt.title(f'Temperature field at t = {Nt*dt:.3} s')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

