from torch.utils.data import Dataset, DataLoader
from collections import deque
import multiprocess as mp
import numpy as np
import torch
import os

def calc_mask_dfs(perm: np.ndarray, sources: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(perm, dtype=bool)
    n, m = mask.shape
    for source in sources:
        source = tuple(source)
        if not perm[source]:
            continue
        mask[source] = True
        q = deque([source])
        while q:
            a, b = q.pop()            
            left_bound = a > 0
            right_bound = a < n - 1
            upper_bound = b > 0
            lower_bound = b < m - 1
            if left_bound:
                x, y = a - 1, b
                if perm[x, y] and not mask[x, y]:
                    mask[x, y] = True
                    q.append((x, y))
            if right_bound:
                x, y = a + 1, b
                if perm[x, y] and not mask[x, y]:
                    mask[x, y] = True
                    q.append((x, y))
            if upper_bound:
                x, y = a, b - 1
                if perm[x, y] and not mask[x, y]:
                    mask[x, y] = True
                    q.append((x, y))
                if left_bound:
                    x, y = a - 1, b - 1
                    if perm[x, y] and not mask[x, y]:
                        mask[x, y] = True
                        q.append((x, y))
                if right_bound:
                    x, y = a + 1, b - 1
                    if perm[x, y] and not mask[x, y]:
                        mask[x, y] = True
                        q.append((x, y))
            if lower_bound:
                x, y = a, b + 1
                q.append((x, y))
                if left_bound:
                    x, y = a - 1, b + 1
                    if perm[x, y] and not mask[x, y]:
                        mask[x, y] = True
                        q.append((x, y))
                if right_bound:
                    x, y = a + 1, b + 1
                    if perm[x, y] and not mask[x, y]:
                        mask[x, y] = True
                        q.append((x, y))
    return mask

class PIImageDataset(Dataset):
    def __init__(self, file_perm: str, file_pressure: str = None, file_sources: str = None, n_jobs: int = 1,
                 tmp_filename=None, force_reload=False):
        if tmp_filename is None:
            tmp_filename = file_perm + '.dataset.pt'
        if os.path.isfile(tmp_filename) and not force_reload:
            print(f'Loading preprocessed dataset from {tmp_filename}...')
            self.perms, self.sources, self.pressures = torch.load(tmp_filename)
        else:
            with open(file_perm, 'rb') as f:
                perms = np.load(f)
            if file_pressure:
                with open(file_pressure, 'rb') as f:
                    pressures = np.load(f)
            else:
                pressures = None
            if file_sources:
                with open(file_sources, 'rb') as f:
                    sources = np.load(f)
            else:
                sources = np.array([((x.shape[0] - 1) // 2, (x.shape[1] - 1) // 2) for x in perms], dtype=int)
                sources = sources.reshape((-1, 1, 2))
            perms, pressures, sources = self.preprocess_data(perms=perms, sources=sources, pressures=pressures, n_jobs=n_jobs)
            self.perms = torch.from_numpy(perms).float()
            self.sources = torch.from_numpy(sources).int()
            if file_pressure:
                self.pressures = torch.from_numpy(pressures).float()
            else:
                self.pressures = None
            torch.save((self.perms, self.sources, self.pressures), tmp_filename)
    
    def preprocess_data(self, perms: np.ndarray, sources: np.ndarray, pressures: np.ndarray = None, 
                        n_jobs: int = 1):
        keep_inds = list()
        with mp.Pool(n_jobs) as p:
            fun = lambda t: calc_mask_dfs(*t)
            for i, mask in enumerate(p.imap(fun, zip(perms, sources))):
                perms[i] *= mask
                if perms[i].std() != 0:
                    keep_inds.append(i)
        perms = perms[keep_inds]
        sources = sources[keep_inds]
        g = 9.812
        mu_oil = 1e-3
        rho_oil = 800
        perms = perms.reshape(perms.shape[0], 1, *perms.shape[1:]) * (g / mu_oil) * 1e-3
        if pressures is not None:
            pressures = pressures[keep_inds][:, 1:] / (g *  rho_oil) * 1e5
        return perms, pressures, sources
    
    def __len__(self):
        return self.perms.shape[0]
    
    def __getitem__(self, index):
        if self.pressures is None:
            return self.perms[index], self.sources[index], None
        return self.perms[index], self.sources[index], self.pressures[index]



# data_folder = 'data'
# file_highres = '1_172.npy'
# file_lowres = '1_470.npy'

# file_lowres_pressure = os.path.join(data_folder, 'perm' + file_lowres)
# lowres_dataset = PIDataset(file_lowres_pressure, n_jobs=12, tmp_filename='data/lowres_dataset.pt', force_reload=True)

# file_highres_pressure = os.path.join(data_folder, 'perm' + file_highres)
# file_highres_perm = os.path.join(data_folder, 'pressure' + file_highres)
# highres_dataset = PIDataset(file_highres_pressure, file_highres_perm, n_jobs=12, tmp_filename='data/highres_dataset.pt', force_reload=True)

