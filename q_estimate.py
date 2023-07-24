
from dataset import PIImageDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

data_folder = 'data'
file_highres = '1_172.npy'
file_lowres = '1_172.npy'

file_highres_pressure = os.path.join(data_folder, 'perm' + file_highres)
file_highres_perm = os.path.join(data_folder, 'pressure' + file_highres)
highres_dataset = PIImageDataset(file_highres_pressure, file_highres_perm, n_jobs=12, tmp_filename='tmp/highres_dataset.pt')

spacing = (50, 25)

press = highres_dataset.pressures / (9.812 *  800) * 1e5
perms = highres_dataset.perms * (9.812 / 1e-3) * 1e-3
# perms_grad_x, perms_grad_y = torch.gradient(highres_dataset.perms, axis=(1, 2), edge_order=2, spacing=spacing)
# # m, n = perms_grad_x.shape[1:]
# # m = np.arange(0, m)
# # n = np.arange(0, n)
# # x, y = np.meshgrid(m, n)
# # norms = (perms_grad_x ** 2 + perms_grad_y ** 2) ** 0.5
# # perms_grad_x /= norms
# # perms_grad_y /= norms
# # plt.figure(dpi=200, figsize=(10,10))
# # plt.quiver(x, y, perms_grad_x[0], perms_grad_y[0], units='xy', scale=1)
# # plt.imshow(highres_dataset.per.sms[0], origin='lower')
# # plt.tight_layout()
# # plt.savefig('test_2.pdf')

# press = press.transpose(0, 1)
# press_dt = torch.gradient(press, dim=0)[0]
# press_dx, press_dy = torch.gradient(press, dim=(2,3))
# press_dxx = torch.gradient(press_dx, dim=2)[0]
# press_dyy = torch.gradient(press_dy, dim=3)[0]

# press_laplacian = press_dxx + press_dyy

# t = press_dx * perms_grad_x + press_dy * perms_grad_y + press_laplacian * perms
S_s = 2e-4

time = 10
sample = 0

from collections import defaultdict
q_norms = defaultdict(list)

# plt.figure(dpi=100, figsize=(30, 4 * 5))
cnt = 0
for i in range(len(perms)):

    for time in range(20):
        p = press[i][:, 0]
        K = perms[i][0]
        
        K_dx, K_dy = torch.gradient(K, dim=(0, 1), edge_order=2, spacing=spacing)
        
        p_dt = torch.gradient(p, dim=0, edge_order=2, spacing=365*24*60*60)[0][time]
        p = p[time]
        p_dx, p_dy = torch.gradient(p, dim=(0, 1), edge_order=2, spacing=spacing)
        p_dxx = torch.gradient(p_dx, dim=0, edge_order=2, spacing=spacing[0])[0]
        p_dyy = torch.gradient(p_dy, dim=1, edge_order=2, spacing=spacing[1])[0]
        laplace = p_dxx + p_dyy
        divk = laplace * K + (p_dx * K_dx) + (p_dy * K_dy)
        divk = torch.gradient(K * p_dx, dim=0, edge_order=2, spacing=spacing[0])[0] + torch.gradient(K * p_dy, dim=1, edge_order=2, spacing=spacing[1])[0]
        
        # divk = torch.gradient(c_perm * press_dx, dim=0, edge_order=2, spacing=spacing)[0] + torch.gradient(c_perm * press_dy, dim=1, edge_order=2, spacing=spacing)[0]
        q = divk - S_s * p_dt
        q_norms[time].append(torch.linalg.norm(q))
    
    if i < 5 or (i == 69) or (i == 146):
        m, n = p_dx.shape
        m = np.arange(0, m)
        n = np.arange(0, n)
        x, y = np.meshgrid(m, n)
        norm_g_p = (p_dx ** 2 + p_dy ** 2) ** 0.5
        norm_g_K = (K_dx ** 2 + K_dy ** 2) ** 0.5
        
        plt.subplot(7, 5, 1 + cnt * 5)
        plt.imshow(K, origin='lower')
        plt.colorbar()
        plt.quiver(x, y, K_dx / norm_g_K, K_dy / norm_g_K, units='xy', scale=1)
        # if not i:
        plt.title(r'$K(x, y)$')
        plt.subplot(7, 5, 2 + cnt * 5)
        plt.imshow(p, origin='lower')
        plt.colorbar()
        plt.quiver(x, y, p_dx / norm_g_p, p_dy / norm_g_p, units='xy', scale=1)
        # if not i:
        plt.title(r'$p(x, y, t=20)$')
        plt.subplot(7, 5, 3 + cnt * 5)
        plt.imshow(p_dt, origin='lower')
        plt.colorbar()
        # if not i:
        plt.title(r'$\frac{\partial}{\partial t} p(x, y)|_{t=20}$')
        plt.subplot(7, 5, 4 + cnt * 5)
        plt.imshow(divk.abs(), origin='lower')
        plt.colorbar()
        # if not i:
        plt.title(r'$|\nabla(K(x, y) \nabla p(x, y, t=20)|$')
        plt.subplot(7, 5, 5 + cnt * 5)
        plt.imshow(q.abs(), origin='lower')
        plt.colorbar()
        plt.title(r'$|q(x, y)|$')
        cnt += 1
plt.tight_layout()
plt.savefig('stf.pdf')