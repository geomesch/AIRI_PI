from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset import PIImageDataset
from models import UNet
from utils import set_seed
from torch import nn
from loss import MaskedMSELoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import os



def get_model(name: str, unet_channels=(128, 64, 32), output_size=(64, 64)):
    if name.lower() == 'unet':
        return UNet(in_channels=1, out_channels=20)
        # return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        #                       in_channels=1, out_channels=20, init_features=32,
        #                       pretrained=False)
    
def compute_multipliers(train_dataset):
    # t, n, m = train_dataset[0][-1].shape
    perm_mult = 1.0
    press_mult = 1.0
    # press_mult = torch.ones(t)
    for perm, _, pressure in train_dataset:
        # mask = perm != 0
        perm_mult = max(perm_mult, perm.max())
        press_mult = max(press_mult, pressure.max())
        # mx = (pressure * mask).reshape(t, -1).max(axis=1)
    return perm_mult, press_mult

def build_figure(K, p, p_pred, use_mask=True):
    K = K.detach().cpu().numpy()
    mask = K != 0
    # if not use_mask:
    #     mask[:] = True
    p = p.detach().cpu().numpy() * mask
    p_pred = p_pred.detach().cpu().numpy() * mask
    f = plt.figure(dpi=150)
    plt.subplot(2, 3, 1)
    plt.imshow(p[0], origin='lower')
    plt.title(r'True $p(x, y, t=1)$')
    plt.subplot(2, 3, 2)
    plt.imshow(p[p.shape[0] // 2], origin='lower')
    plt.title(r'True $p(x, y, t=10)$')
    plt.subplot(2, 3, 3)
    plt.imshow(p[-1], origin='lower')
    plt.title(r'True $p(x, y, t=20)$')
    plt.subplot(2, 3, 4)
    plt.imshow(p_pred[0], origin='lower')
    plt.title(r'Predicted $p(x, y, t=1)$')
    plt.subplot(2, 3, 5)
    plt.imshow(p_pred[p.shape[0] // 2], origin='lower')
    plt.title(r'Predicted $p(x, y, t=10)$')
    plt.subplot(2, 3, 6)
    plt.imshow(p_pred[-1], origin='lower')
    plt.title(r'Predicted $p(x, y, t=20)$')
    plt.tight_layout()
    return f

apply_masking = True
train_fraction = 0.8
batch_size = 16
num_workers = 10
summary_dir = 'tmp'
comment = 'mask'
N_epoch = 500
plot_step = 20
seed = 1


set_seed(seed)

model = get_model('unet').cuda()

data_folder = 'data'
file_highres = '1_172.npy'
file_lowres = '1_470.npy'

file_lowres_pressure = os.path.join(data_folder, 'pressure' + file_lowres)
file_lowres_perm = os.path.join(data_folder, 'perm' + file_lowres)

dataset = PIImageDataset(file_lowres_perm, file_lowres_pressure, n_jobs=num_workers, tmp_filename='tmp/lowres_dataset.pt')

train_size = int(train_fraction * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
dl_test = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=num_workers)

perm_mult, press_mult = compute_multipliers(train_dataset)

loss = MaskedMSELoss()

optimizer = torch.optim.Adam(model.parameters())

os.makedirs(summary_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(summary_dir, comment), comment=comment)

for epoch in tqdm(list(range(500))):
    model.train(True)
    train_loss = 0.0
    for K, sources, p in dl_train:
        K = K.cuda()
        p = p.cuda()
        with torch.no_grad():
            K = K / perm_mult
            p = p / press_mult
        optimizer.zero_grad()
        p_pred = model(K)
        if apply_masking:
            lf = loss(p_pred, p, K != 0)
        else:
            lf = loss(p_pred, p)
        lf.backward()
        optimizer.step()
        train_loss += lf
    writer.add_scalar('Loss/train', train_loss, epoch)
    model.train(False)
    test_loss_masked = 0.0
    test_loss = 0.0
    for K, sources, p in dl_test:
        K = K.cuda() / perm_mult
        p = p.cuda() / press_mult
        with torch.no_grad():
            p_pred = model(K)
            test_loss_masked += loss(p_pred, p, K != 0)
            test_loss += loss(p_pred, p)
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Loss/test_masked', test_loss_masked, epoch)
    if not epoch % plot_step:
        with torch.no_grad():
            K, _, p = test_dataset[0]
            p = p / press_mult
            K = K.unsqueeze(0).cuda() / perm_mult
            p_pred = model(K)[0]
            writer.add_figure('train/fig', build_figure(K[0,0], p, p_pred, use_mask=apply_masking), epoch)
            K, _, p = test_dataset[0]
            p = p / press_mult
            K = K.unsqueeze(0).cuda() / perm_mult
            p_pred = model(K)[0]
            writer.add_figure('test/fig', build_figure(K[0,0], p, p_pred, use_mask=apply_masking), epoch)
writer.flush()
writer.close()