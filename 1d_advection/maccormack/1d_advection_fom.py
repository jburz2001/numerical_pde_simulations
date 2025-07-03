import numpy as np
import matplotlib.pyplot as plt
import h5py

def get_fwd_diff_op(u, dx):
    n = u.shape[0]
    main_diag  = -1*np.ones(n)
    super_diag = np.ones(n-1)
    K = np.diag(main_diag) + np.diag(super_diag, k=1)
    K[-1,-1] = 1
    K[-1,-2] = -1
    K *= (1/dx)
    return K

def get_bwd_diff_op(u, dx):
    n = u.shape[0]
    main_diag = np.ones(n)
    sub_diag  = -1*np.ones(n-1)
    K = np.diag(main_diag) + np.diag(sub_diag, k=-1)
    K[0,0] = -1
    K[0,1] = 1
    K *= (1/dx)
    return K

def get_u_next(u_curr, a, dt, Kfwd, Kbwd):
    u_pred = u_curr - (a*dt)*(Kfwd@u_curr)
    return ((u_curr + u_pred)/2) - (a*dt/2)*(Kbwd@u_pred)

def get_u0(x, mu):
    return  np.exp(- ((x - mu)**2)/(0.0002)) / np.sqrt(0.0002*np.pi)

if __name__ == '__main__':
    x0, xf = 0, 1
    n = 2**12
    x = np.linspace(x0, xf, num=n, endpoint=True)
    dx = x[1] - x[0]

    t0, tf = 0, 0.08
    c  = 10
    dt = dx / c
    nt = int(np.ceil((tf-t0)/dt))
    ts  = np.zeros(nt+1)
    ts[0] = t0

    U = np.zeros((x.shape[0], nt+1))
    mu = 0.12547
    u_curr = get_u0(x, mu)
    U[:,0] = u_curr

    Kfwd = get_fwd_diff_op(u_curr, dx)
    Kbwd = get_bwd_diff_op(u_curr, dx)

    t_curr = t0
    for j in range(1, nt+1):
        if (j % 100) == 0:
            print(t_curr)

        # if t_curr + dt > tf:
        #     dt = tf - t_curr

        u_next = get_u_next(u_curr, c, dt, Kfwd, Kbwd)
        u_curr = u_next
        U[:, j] = u_curr
        
        t_curr += dt
        ts[j] = t_curr

    with h5py.File('1d_advection_fom.h5', 'w') as f:
        f.create_dataset('snapshots', data=U)
        f.create_dataset('x_domain', data=x)
        f.create_dataset('timestamps', data=ts)
        f.create_dataset('advection_speed', data=c)
        f.create_dataset('mean_of_initial_pulse', data=mu)
        f.create_dataset('number_of_discretization_DOFs', data=n)


    # example plot at initial time
    plt.plot(x, U[:, 0])
    plt.xlabel('x')
    plt.ylabel(f'u(x, ts = {t0})')
    plt.ylim((-1,40))
    plt.title(f'1D Advection, MacCormack at ts={t0}')
    plt.show()

    # example plot at final time
    plt.plot(x, U[:, -1])
    plt.xlabel('x')
    plt.ylabel(f'u(x, ts = {t_curr})')
    plt.ylim((-1,40))
    plt.title(f'1D Advection, MacCormack at ts={t_curr}')
    plt.show()

